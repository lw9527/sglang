from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.alignment import ALIGNMENT_BLOCK_2M
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)
from sglang.srt.utils import get_bool_env_var


def _npu_padded_page_count(
    num_pages: int, page_size: int, head_dim: int, dtype: torch.dtype
) -> int:
    """Pad per-layer KV slab so layer strides are 2 MiB-aligned (PD IPC + dim_exchange)."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    layer_bytes = num_pages * page_size * 1 * head_dim * elem_size
    padded_layer_bytes = (
        (layer_bytes + ALIGNMENT_BLOCK_2M - 1) // ALIGNMENT_BLOCK_2M
    ) * ALIGNMENT_BLOCK_2M
    page_bytes = page_size * 1 * head_dim * elem_size
    return padded_layer_bytes // page_bytes


if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # 2 MiB aligned via base_addr_aligned_kb=2048.
            segment_shape = (
                self.size // self.page_size + 1,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
            self.k_buffer = [
                torch.zeros(segment_shape, dtype=self.store_dtype, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(segment_shape, dtype=self.store_dtype, device=self.device)
                for _ in range(self.layer_num)
            ]

            if self.use_fia:
                self.k_buffer = [
                    s.view(-1, 1, self.head_num, self.head_dim) for s in self.k_buffer
                ]
                self.v_buffer = [
                    s.view(-1, 1, self.head_num, self.head_dim) for s in self.v_buffer
                ]

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    # so indexing with flat token-slot ids only lines up when page_size == 1.
    # Override to flatten explicitly so both layouts (and page_size > 1) work.
    def get_cpu_copy(self, indices):
        torch.npu.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = k_layer[chunk_indices].to("cpu", non_blocking=True)
                v_cpu = v_layer[chunk_indices].to("cpu", non_blocking=True)
                layer_chunks.append([k_cpu, v_cpu])
            kv_cache_cpu.append(layer_chunks)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]

            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer,
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer,
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.head_num, self.head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                slot_indices=loc,
            )


class NPUMLATokenToKVPool(MLATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # 2 MiB aligned via base_addr_aligned_kb=2048.
            # 5D layout (layer, page, page_size, 1, dim) for transfer_kv_dim_exchange;
            # pad the page dim so each layer base is 2 MiB-aligned for PD IPC.
            self._num_pages = self.size // self.page_size + 1
            k_padded_pages = _npu_padded_page_count(
                self._num_pages, self.page_size, self.kv_lora_rank, self.store_dtype
            )
            v_padded_pages = _npu_padded_page_count(
                self._num_pages,
                self.page_size,
                self.qk_rope_head_dim,
                self.store_dtype,
            )
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    k_padded_pages,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    v_padded_pages,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                ik_padded_pages = _npu_padded_page_count(
                    self._num_pages,
                    self.page_size,
                    self.index_head_dim,
                    self.store_dtype,
                )
                self.index_k_buffer = torch.zeros(
                    (
                        self.layer_num,
                        ik_padded_pages,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)

    def _kv_layer(self, buf: torch.Tensor, local_layer_id: int) -> torch.Tensor:
        """Logical KV pages only; trailing padding is for 2 MiB layer stride / HiCache."""
        return buf[local_layer_id, : self._num_pages]

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = get_tensor_size_bytes(self.k_buffer) + get_tensor_size_bytes(
            self.v_buffer
        )
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            kv_size_bytes += get_tensor_size_bytes(self.index_k_buffer)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        local_id = layer_id - self.start_layer
        return (
            self._kv_layer(self.k_buffer, local_id),
            self._kv_layer(self.v_buffer, local_id),
        )

    def get_state_buf_infos(self):
        # NOTE: index_k_buffer is also exposed in get_contiguous_buf_infos()
        # below (kv_data_ptrs[2*N:3*N]). PD-disagg callers (prefill.py /
        # decode.py) explicitly SKIP this method for NPUMLATokenToKVPool to
        # avoid registering the same buffer twice with Mooncake (the
        # second batch_register would be rejected as exact_dup overlap and
        # leave the underlying ADXL/HIXL engine in an inconsistent state,
        # surfacing later as connect status 503900). This method is kept
        # for non-PD callers that want to introspect the indexer cache.
        data_ptrs = [
            self._kv_layer(self.index_k_buffer, i).data_ptr()
            for i in range(self.layer_num)
        ]
        data_lens = [
            self._kv_layer(self.index_k_buffer, i).nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self._kv_layer(self.index_k_buffer, i)[0].nbytes
            for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        buf = self._kv_layer(self.k_buffer, layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return buf.view(self.dtype)
        return buf

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        buf = self._kv_layer(self.v_buffer, layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return buf.view(self.dtype)
        return buf

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        buf = self._kv_layer(self.index_k_buffer, layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return buf.view(self.dtype)
        return buf

    # for disagg
    def get_contiguous_buf_infos(self):
        # NPU MLA / NSA layout. K/V/index_k are 5D tensors (layer, page, ...);
        # per-layer views share one allocation with 2 MiB-padded layer stride.
        # The returned lists are GROUP-ORDERED:
        #     kv_data_ptrs = [K_0..K_{N-1}, V_0..V_{N-1}, IK_0..IK_{N-1}]
        #     kv_item_lens = [k_il...,      v_il...,      ik_il...      ]
        # where each *_il may differ (head dims differ for K/V/IK in NSA).
        # PD-disagg callers must treat this as 2 or 3 GROUPS per layer, NOT
        # a single layer-stripe list. See:
        #   - common/conn.py:get_mla_kv_ptrs_with_pp (group-aware PP slicing)
        #   - mooncake/conn.py:_send_kvcache_generic (per-entry item_len)
        kv_data_ptrs = [
            self._kv_layer(self.k_buffer, i).data_ptr() for i in range(self.layer_num)
        ] + [self._kv_layer(self.v_buffer, i).data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [
            self._kv_layer(self.k_buffer, i).nbytes for i in range(self.layer_num)
        ] + [self._kv_layer(self.v_buffer, i).nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self._kv_layer(self.k_buffer, i)[0].nbytes for i in range(self.layer_num)
        ] + [self._kv_layer(self.v_buffer, i)[0].nbytes for i in range(self.layer_num)]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self._kv_layer(self.index_k_buffer, i).data_ptr()
                for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self._kv_layer(self.index_k_buffer, i).nbytes
                for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self._kv_layer(self.index_k_buffer, i)[0].nbytes
                for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        local_id = layer_id - self.start_layer
        k_layer = self._kv_layer(self.k_buffer, local_id)
        v_layer = self._kv_layer(self.v_buffer, local_id)

        torch_npu.npu_scatter_nd_update_(
            k_layer.view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            v_layer.view(-1, 1, self.qk_rope_head_dim),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self._kv_layer(self.index_k_buffer, layer_id - self.start_layer).view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    # The parent MLATokenToKVPool stores K/V combined in `self.kv_buffer` and
    # offloads via that single tensor. NPU MLA / NSA splits them into separate
    # k_buffer / v_buffer (and optionally index_k_buffer), each shaped
    # (num_pages, page_size, 1, dim), so the parent implementations crash with
    # `AttributeError: ... has no attribute 'kv_buffer'` when
    # `retract_decode -> offload_kv_cache` runs. Override to offload each
    # buffer independently using the same flat token-slot indexing as
    # set_kv_buffer / set_index_k_buffer (page_size > 1 must work).
    def get_cpu_copy(self, indices):
        torch.npu.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self._kv_layer(self.k_buffer, local_layer_id).view(
                -1, 1, self.kv_lora_rank
            )
            v_layer = self._kv_layer(self.v_buffer, local_layer_id).view(
                -1, 1, self.qk_rope_head_dim
            )
            ik_layer = (
                self._kv_layer(self.index_k_buffer, local_layer_id).view(
                    -1, 1, self.index_head_dim
                )
                if has_ik
                else None
            )
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = k_layer[chunk_indices].to("cpu", non_blocking=True)
                v_cpu = v_layer[chunk_indices].to("cpu", non_blocking=True)
                if has_ik:
                    ik_cpu = ik_layer[chunk_indices].to("cpu", non_blocking=True)
                    layer_chunks.append((k_cpu, v_cpu, ik_cpu))
                else:
                    layer_chunks.append((k_cpu, v_cpu))
            kv_cache_cpu.append(layer_chunks)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self._kv_layer(self.k_buffer, local_layer_id).view(
                -1, 1, self.kv_lora_rank
            )
            v_layer = self._kv_layer(self.v_buffer, local_layer_id).view(
                -1, 1, self.qk_rope_head_dim
            )
            ik_layer = (
                self._kv_layer(self.index_k_buffer, local_layer_id).view(
                    -1, 1, self.index_head_dim
                )
                if has_ik
                else None
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if has_ik:
                    ik_cpu = chunk[2]
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
        torch.npu.synchronize()
