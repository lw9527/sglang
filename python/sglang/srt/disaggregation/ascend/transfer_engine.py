import logging
import os
from typing import List, Optional

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
        store_url: Optional[str] = None,
    ):
        """Create one Ascend MemFabric TransferEngine bound to a single store.

        Args:
            hostname: Local hostname/IP used to compose the engine ``unique_id``.
            npu_id: NPU device id passed straight to MemFabric.
            disaggregation_mode: PREFILL or DECODE; determines the ``role``.
            store_url: MemFabric config store URL this engine should register
                with. If ``None`` we fall back to the legacy ``ASCEND_MF_STORE_URL``
                environment variable purely for backward-compatibility with old
                deployment scripts; new code paths should always pass an
                explicit per-Prefill store_url to avoid the SPOF where every
                P/D depended on the first P's address.
        """
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/backend/pd_disaggregation.md"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        if store_url is None:
            store_url = os.getenv("ASCEND_MF_STORE_URL")
            if store_url:
                logger.warning(
                    "AscendTransferEngine falling back to ASCEND_MF_STORE_URL=%s. "
                    "This still pins all peers to a single global store and is "
                    "kept only for legacy compatibility.",
                    store_url,
                )
        if not store_url:
            raise ValueError(
                "AscendTransferEngine requires a store_url. Either pass it "
                "explicitly (preferred, per-Prefill self-host) or set "
                "ASCEND_MF_STORE_URL for legacy deployments."
            )
        self.store_url = store_url
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()
        self.initialize()

    @property
    def unique_id(self) -> str:
        """Alias for the MemFabric engine ``unique_id`` (== session_id, ip:port)."""
        return self.session_id

    @staticmethod
    def derive_local_store_url(local_ip: str, port: int) -> str:
        """Compose the ``tcp://<local_ip>:<port>`` URL each Prefill instance
        should use to self-host its own MemFabric config store."""
        return f"tcp://{local_ip}:{int(port)}"

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
        )
        if ret_value != 0:
            logger.error(
                f"Ascend Transfer Engine initialization failed for store_url={self.store_url}."
            )
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "sdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
