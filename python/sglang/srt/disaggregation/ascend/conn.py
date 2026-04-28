import concurrent.futures
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    """KVManager for the Ascend NPU backend.

    Architecture (per-Prefill self-hosted MemFabric store):
      * Each Prefill instance brings up its **own** MemFabric ``config_store``
        on ``tcp://<local_ip>:<bootstrap_port + 1>`` and a single
        ``TransferEngine`` bound to it. The store_url + engine unique_id are
        advertised through the SGLang HTTP bootstrap server so any Decode can
        discover them.
      * Each Decode keeps a lazy pool of ``TransferEngine`` instances, one per
        discovered Prefill ``store_url``. When a request first targets a P,
        the corresponding engine is created and the local KV/aux/state buffers
        are registered with it. The Decode's ``session_id`` reported back to a
        given P is the session_id of the engine bound to **that** P's store.
      * If the legacy ``ASCEND_MF_STORE_URL`` env var is set we fall back to
        the old "single shared store" topology for backward-compat. In that
        case every P uses the same store_url so the D-side pool naturally
        collapses to a single shared engine.

    Removing the global ASCEND_MF_STORE_URL eliminates the Single Point of
    Failure where killing P0 (the host of the shared store) brought down the
    entire P/D fleet, matching the GPU/Mooncake decentralized rendezvous.
    """

    def __init__(
        self,
        args,
        disaggregation_mode,
        server_args,
        is_mla_backend: Optional[bool] = False,
    ):
        # Compute store strategy upfront so init_engine() (called from super)
        # can use it.
        self._compute_local_store_url(server_args, disaggregation_mode)
        # Lazy engine pool used on the Decode side. Initialized here so it
        # exists before any helper method may inspect it.
        self.engines: Dict[str, AscendTransferEngine] = {}
        self.engines_lock = threading.Lock()
        # CommonKVManager.__init__ (called from super) registers the Prefill
        # to the bootstrap server BEFORE MooncakeKVManager.__init__ runs
        # init_engine(). Our _get_extra_bootstrap_payload needs ``self.engine``
        # to be live, so defer the registration through this flag and
        # re-trigger it manually at the bottom of __init__.
        self._defer_bootstrap_registration = True
        try:
            super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        finally:
            self._defer_bootstrap_registration = False
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.register_to_bootstrap()

    def register_to_bootstrap(self):
        # Skip the bootstrap POST while CommonKVManager.__init__ is still
        # running (engine not initialized yet). __init__ will call us again
        # after super() returns and init_engine() has produced self.engine.
        if getattr(self, "_defer_bootstrap_registration", False):
            return
        super().register_to_bootstrap()

    # ------------------------------------------------------------------
    # Store / engine setup
    # ------------------------------------------------------------------

    def _compute_local_store_url(self, server_args, disaggregation_mode) -> None:
        """Decide which store_url this process should use.

        * Prefill + new mode: ``tcp://<local_ip>:<bootstrap_port + 1>``. The
          MemFabric store port is always derived from
          ``--disaggregation-bootstrap-port + 1`` so users only need to manage
          a single port for both the SGLang HTTP rendezvous and the per-P
          MemFabric store.
        * Legacy mode (``ASCEND_MF_STORE_URL`` set): use the env var verbatim.
        * Decode: ``local_store_url`` is unused (engines are created lazily
          against each P's advertised store_url) but we still record what env
          var was set, so receivers can fall back to the legacy single-store
          path when bootstrap_info lacks ``store_url``.
        """
        legacy_url = os.getenv("ASCEND_MF_STORE_URL")
        self._using_legacy_global_store = bool(legacy_url)
        if disaggregation_mode == DisaggregationMode.PREFILL:
            if legacy_url:
                self.local_store_url = legacy_url
            else:
                store_port = server_args.disaggregation_bootstrap_port + 1
                self.local_store_url = AscendTransferEngine.derive_local_store_url(
                    get_local_ip_auto(), store_port
                )
        else:
            # Decode: nothing to host.
            self.local_store_url = legacy_url

    def init_engine(self):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            local_ip = get_local_ip_auto()
            self.engine = AscendTransferEngine(
                hostname=local_ip,
                npu_id=self.kv_args.gpu_id,
                disaggregation_mode=self.disaggregation_mode,
                store_url=self.local_store_url,
            )
            logger.info(
                "Ascend Prefill TransferEngine initialized, store_url=%s, "
                "session_id=%s",
                self.local_store_url,
                self.engine.session_id,
            )
        else:
            # Decode: lazy engine pool, no engine yet. ``self.engine`` is kept
            # as ``None`` so callers can detect the lazy mode.
            self.engine = None

    def register_buffer_to_engine(self):
        if self.engine is not None:
            self._register_buffer_to_engine_impl(self.engine)

    def _register_buffer_to_engine_impl(self, engine: AscendTransferEngine) -> None:
        """Register all known KV / aux / state buffers with one engine.
        Used both by the eager Prefill path and by Decode's lazy pool."""
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimizes batch registration for small blocks.
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        start_layer = self.kv_args.prefill_start_layer
        if self.kv_args.state_type == "nsa":
            src_layers = len(src_kv_ptrs) // 3
            total_layers = len(dst_kv_ptrs) // 3
            end_layer = start_layer + src_layers
            if src_layers == total_layers:
                sliced_dst_kv_ptrs = dst_kv_ptrs
            else:
                k_ptrs = dst_kv_ptrs[start_layer:end_layer]
                v_ptrs = dst_kv_ptrs[
                    total_layers + start_layer : total_layers + end_layer
                ]
                index_k_ptrs = dst_kv_ptrs[
                    2 * total_layers + start_layer : 2 * total_layers + end_layer
                ]
                sliced_dst_kv_ptrs = k_ptrs + v_ptrs + index_k_ptrs
        else:
            src_layers = len(src_kv_ptrs) // 2
            total_layers = len(dst_kv_ptrs) // 2
            end_layer = start_layer + src_layers
            if src_layers == total_layers:
                sliced_dst_kv_ptrs = dst_kv_ptrs
            else:
                k_ptrs = dst_kv_ptrs[start_layer:end_layer]
                v_ptrs = dst_kv_ptrs[
                    total_layers + start_layer : total_layers + end_layer
                ]
                sliced_dst_kv_ptrs = k_ptrs + v_ptrs

        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def get_or_create_engine_for_prefill(self, store_url: str) -> AscendTransferEngine:
        """Return (lazily creating if needed) the Decode-side engine bound to
        ``store_url`` so it can talk to whichever Prefill instance hosts that
        store. Idempotent and thread-safe."""
        assert (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ), "get_or_create_engine_for_prefill is only meaningful on the Decode side"
        if not store_url:
            raise ValueError(
                "Empty store_url passed to get_or_create_engine_for_prefill"
            )
        with self.engines_lock:
            engine = self.engines.get(store_url)
            if engine is None:
                logger.info(
                    "Creating new Ascend Decode engine for prefill store_url=%s",
                    store_url,
                )
                engine = AscendTransferEngine(
                    hostname=get_local_ip_auto(),
                    npu_id=self.kv_args.gpu_id,
                    disaggregation_mode=self.disaggregation_mode,
                    store_url=store_url,
                )
                self._register_buffer_to_engine_impl(engine)
                self.engines[store_url] = engine
            return engine

    def get_session_id(self):
        # Prefill side has a single engine.
        if self.engine is not None:
            return self.engine.get_session_id()
        # Decode side new mode: per-P engine is chosen lazily; receivers must
        # use ``_session_id_for(bootstrap_info)`` to pick the right one.
        return ""

    def _get_extra_bootstrap_payload(self) -> Dict[str, Optional[str]]:
        """Inject this Prefill's store_url + engine unique_id into the
        bootstrap payload so Decode side can discover them."""
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return {}
        return {
            "store_url": self.local_store_url,
            "engine_unique_id": self.engine.unique_id,
        }

    # ------------------------------------------------------------------
    # KV transfer (Prefill side)
    # ------------------------------------------------------------------

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    """Decode-side receiver. Picks the per-P engine session_id when sending
    its identity to a Prefill so the destflag is resolvable in that P's
    MemFabric store."""

    def _session_id_for(self, bootstrap_info: dict) -> str:
        store_url = bootstrap_info.get("store_url") if bootstrap_info else None
        if not store_url:
            # Legacy / non-ascend bootstrap entry → fall back to the receiver's
            # default session_id (which on Ascend is "" in new mode and the
            # legacy global engine's id when ASCEND_MF_STORE_URL is set).
            return self.session_id
        engine = self.kv_mgr.get_or_create_engine_for_prefill(store_url)
        return engine.session_id


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
