"""Diagnostics for NPU HCCL IPC 2MB alignment (PD / Mooncake ascend protocol)."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

NPU_IPC_PAGE_SIZE_2MB = 2097152


def ipc_region_aligned(ptr: int, length: int) -> bool:
    page = NPU_IPC_PAGE_SIZE_2MB
    return ptr % page == 0 and length % page == 0 and length > 0


def should_log_npu_ipc_alignment() -> bool:
    if envs.SGLANG_NPU_IPC_ALIGN_DEBUG.get():
        return True
    try:
        from sglang.srt.utils.common import is_npu

        return is_npu()
    except RuntimeError:
        return False


def log_ipc_regions(
    source: str,
    ptrs: Sequence[int],
    lengths: Sequence[int],
    labels: Optional[Sequence[str]] = None,
) -> int:
    """Log 2MB alignment status. Returns count of misaligned regions.

    Emits WARNING for each misaligned buffer on NPU (or when
    SGLANG_NPU_IPC_ALIGN_DEBUG=1). Emits INFO for all buffers when debug is on.
    """
    if not should_log_npu_ipc_alignment():
        return 0

    page = NPU_IPC_PAGE_SIZE_2MB
    debug_all = envs.SGLANG_NPU_IPC_ALIGN_DEBUG.get()
    misaligned = 0
    n = len(ptrs)

    for i in range(n):
        ptr = int(ptrs[i])
        length = int(lengths[i])
        label = labels[i] if labels is not None and i < len(labels) else f"buffer[{i}]"
        ptr_ok = ptr % page == 0
        len_ok = length % page == 0
        if ptr_ok and len_ok:
            if debug_all:
                logger.info(
                    "[NPU IPC align] %s %s: ptr=0x%x size=%d OK",
                    source,
                    label,
                    ptr,
                    length,
                )
            continue

        misaligned += 1
        ptr_rem = ptr % page
        len_rem = length % page
        logger.warning(
            "[NPU IPC align] %s %s: ptr=0x%x (offset=%d) size=%d (remainder=%d) "
            "— NOT 2MB aligned (page=%d); HCCL IPC may fail on this region",
            source,
            label,
            ptr,
            ptr_rem,
            length,
            len_rem,
            page,
        )

    if misaligned:
        logger.warning(
            "[NPU IPC align] %s: %d/%d region(s) misaligned",
            source,
            misaligned,
            n,
        )
    elif debug_all and n:
        logger.info(
            "[NPU IPC align] %s: all %d region(s) are 2MB aligned",
            source,
            n,
        )
    return misaligned
