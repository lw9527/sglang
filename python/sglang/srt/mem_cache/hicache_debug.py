"""Debug logging for HiCache paths before and during prefill scheduling.

Enable with: export SGLANG_DEBUG_HICACHE=1

Logs use WARNING level so they appear with default server logging.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

import torch.distributed as dist

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Throttle repeated scheduler poll logs (same checkpoint + unchanged state).
_THROTTLE_INTERVAL_S = 5.0
_last_throttled: dict[str, tuple[float, str]] = {}


def hicache_debug_enabled() -> bool:
    return envs.SGLANG_DEBUG_HICACHE.get()


def rank_tag_from_cache(cache: Any) -> str:
    tp_rank = -1
    try:
        if getattr(cache, "tp_group", None) is not None:
            tp_rank = dist.get_rank(group=cache.tp_group)
    except Exception:
        pass
    return (
        f"PP{getattr(cache, 'pp_rank', '?')}"
        f" TP{tp_rank}"
        f" CP{getattr(cache, 'attn_cp_rank', '?')}"
    )


def _seq_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        numel = getattr(value, "numel", None)
        if callable(numel):
            return int(numel())
        return len(value)
    except Exception:
        return 0


def req_brief(req: Any) -> str:
    if req is None:
        return ""
    try:
        rid = getattr(req, "rid", "")
        extend_len = getattr(req, "extend_input_len", None)
        host_hit = getattr(req, "host_hit_length", None)
        prefix_len = _seq_len(getattr(req, "prefix_indices", None))
        fill_len = _seq_len(getattr(req, "fill_ids", None))
        return (
            f"rid={rid} extend={extend_len} host_hit={host_hit} "
            f"prefix={prefix_len} fill={fill_len}"
        )
    except Exception as exc:
        return f"rid={getattr(req, 'rid', '?')} brief_error={exc!r}"


def is_hicache_idle(snapshot: dict[str, Any]) -> bool:
    return all(
        snapshot.get(k, 0) == 0
        for k in (
            "ongoing_write_through",
            "ongoing_load_back",
            "ongoing_prefetch",
            "ongoing_backup",
            "ack_write_q",
            "ack_load_q",
            "write_q",
            "load_q",
        )
    )


def log_hicache(
    event: str,
    *,
    cache: Any = None,
    rank_tag: str = "",
    req_id: str = "",
    throttle_key: str = "",
    **fields: Any,
) -> None:
    if not hicache_debug_enabled():
        return
    if throttle_key:
        state = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        now = time.monotonic()
        prev = _last_throttled.get(throttle_key)
        if prev is not None:
            last_t, last_state = prev
            if now - last_t < _THROTTLE_INTERVAL_S and last_state == state:
                return
        _last_throttled[throttle_key] = (now, state)

    tag = rank_tag or (rank_tag_from_cache(cache) if cache is not None else "")
    parts = ["[HICACHE_DEBUG]", event]
    if tag:
        parts.append(tag)
    if req_id:
        parts.append(f"rid={req_id}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.warning(" ".join(parts))


def snapshot_hicache_state(cache: Any) -> dict[str, Any]:
    cc = getattr(cache, "cache_controller", None)
    out: dict[str, Any] = {
        "ongoing_write_through": len(getattr(cache, "ongoing_write_through", {}) or {}),
        "ongoing_load_back": len(getattr(cache, "ongoing_load_back", {}) or {}),
        "ongoing_prefetch": len(getattr(cache, "ongoing_prefetch", {}) or {}),
        "ongoing_backup": len(getattr(cache, "ongoing_backup", {}) or {}),
    }
    if cc is not None:
        # out["prefetch_revoke_q"] = cc.prefetch_revoke_queue.qsize()
        out["ack_write_q"] = len(getattr(cc, "ack_write_queue", []) or [])
        out["ack_load_q"] = len(getattr(cc, "ack_load_queue", []) or [])
        out["write_q"] = len(getattr(cc, "write_queue", []) or [])
        out["load_q"] = len(getattr(cc, "load_queue", []) or [])
        out["prefetch_tokens_occupied"] = getattr(cc, "prefetch_tokens_occupied", None)
    return out


@contextmanager
def log_hicache_block(
    event: str,
    *,
    cache: Any = None,
    req_id: str = "",
    slow_ms: float = 50.0,
    **fields: Any,
):
    """Log enter/exit around a potentially blocking HiCache call."""
    if not hicache_debug_enabled():
        yield
        return
    t0 = time.monotonic()
    log_hicache(f"{event}_ENTER", cache=cache, req_id=req_id, **fields)
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        exit_fields = dict(fields)
        exit_fields["elapsed_ms"] = f"{elapsed_ms:.1f}"
        level_event = event
        if elapsed_ms >= slow_ms:
            level_event = f"{event}_SLOW"
        log_hicache(f"{level_event}_EXIT", cache=cache, req_id=req_id, **exit_fields)


def log_event_sync(
    event: str,
    finish_event: Any,
    *,
    cache: Any = None,
    req_id: str = "",
    **fields: Any,
) -> None:
    """Log around device event synchronize (common hang point on NPU)."""
    if not hicache_debug_enabled():
        finish_event.synchronize()
        return
    queried = False
    try:
        queried = finish_event.query()
    except Exception:
        pass
    log_hicache(
        f"{event}_SYNC",
        cache=cache,
        req_id=req_id,
        event_queried=queried,
        **fields,
    )
    t0 = time.monotonic()
    finish_event.synchronize()
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    log_hicache(
        f"{event}_SYNC_DONE",
        cache=cache,
        req_id=req_id,
        elapsed_ms=f"{elapsed_ms:.1f}",
        **fields,
    )
