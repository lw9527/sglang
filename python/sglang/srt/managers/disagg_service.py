"""Start bootstrap/kv-store-related server"""

import logging
import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def start_disagg_service(
    server_args: ServerArgs,
):
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        # Each Prefill instance hosts its own MemFabric config store on the
        # ``node_rank == 0`` sub-process of that instance. Sub-ranks within
        # the same instance still talk to this local store (legacy multi-node
        # P semantics preserved). Different P instances no longer share a
        # single store, removing the SPOF where killing the cluster's first P
        # took everyone down.
        if transfer_backend == TransferBackend.ASCEND and server_args.node_rank == 0:
            _start_ascend_local_config_store(server_args)

        return bootstrap_server


def _start_ascend_local_config_store(server_args: ServerArgs) -> str:
    """Bring up a MemFabric ``create_config_store`` for this Prefill instance.

    Resolution order for the store URL:

    1. ``ASCEND_MF_STORE_URL`` (if explicitly set): legacy "single global store"
       mode is preserved -- the operator is responsible for making sure only
       one cluster-wide P process owns that address. Still a SPOF, but kept
       for backward compatibility with existing deployment scripts.
    2. Otherwise derive ``tcp://<local_ip>:<disaggregation_bootstrap_port + 1>``
       so each Prefill instance is self-hosted and independent. The store port
       is intentionally tied to the bootstrap port so users only have to
       reason about a single ``--disaggregation-bootstrap-port`` knob.
    """
    try:
        from memfabric_hybrid import create_config_store
    except ImportError as e:
        raise RuntimeError(
            "memfabric_hybrid is required for the Ascend disaggregation backend"
        ) from e

    legacy_url = os.getenv("ASCEND_MF_STORE_URL")
    if legacy_url:
        store_url = legacy_url
        logger.warning(
            "Using legacy ASCEND_MF_STORE_URL=%s for MemFabric config store. "
            "All Prefill/Decode peers will share this single address, which "
            "remains a single point of failure. Unset the env var to enable "
            "the per-Prefill self-hosted store mode.",
            store_url,
        )
    else:
        from sglang.srt.utils.network import get_local_ip_auto

        store_port = server_args.disaggregation_bootstrap_port + 1
        store_url = f"tcp://{get_local_ip_auto()}:{store_port}"
        logger.info("Starting per-Prefill MemFabric config store at %s", store_url)

    try:
        create_config_store(store_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create MemFabric config store at {store_url}: {e}"
        ) from e

    return store_url
