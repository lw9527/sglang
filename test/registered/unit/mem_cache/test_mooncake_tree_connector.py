import threading
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_mappings import (
    DevicePoolEntry,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_tree_connector
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_tree_connector import (
    LayerWiseLoadCounter,
    MooncakeTreeConnector,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ConnectorTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_connector_mixin import (
    UnifiedCacheConnectorMixin,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots=None):
        self.slots = slots
        self.freed = []
        self.mapping = []

    def available_size(self):
        return 100

    def alloc(self, size):
        if self.slots is None:
            return torch.arange(1, size + 1, dtype=torch.int64)
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def test_connector_tail_hashes_follow_radix_key_semantics():
    mixin = UnifiedCacheConnectorMixin()
    mixin.page_size = 2
    no_device_hit = SimpleNamespace(last_device_node=None)

    regular = RadixKey([1, 2, 3, 4])
    regular_hashes = mixin._connector_tail_keys(regular, no_device_hit, 0)
    first_regular = regular.hash_page(0, 2)
    assert regular_hashes == [
        first_regular,
        regular.hash_page(2, 4, first_regular),
    ]

    # A bigram page hashes overlapping pairs, not a plain raw-token slice.
    bigram = RadixKey([1, 2, 3, 4, 5], is_bigram=True)
    first_bigram = bigram.hash_page(0, 2)
    anchored = SimpleNamespace(
        last_device_node=SimpleNamespace(
            get_last_hash_value=lambda: first_bigram,
        )
    )
    assert mixin._connector_tail_keys(bigram, anchored, 2) == [
        bigram.hash_page(2, 4, first_bigram)
    ]

    salted_a = RadixKey([1, 2], extra_key="adapter-a")
    salted_b = RadixKey([1, 2], extra_key="adapter-b")
    assert mixin._connector_tail_keys(salted_a, no_device_hit, 0) != (
        mixin._connector_tail_keys(salted_b, no_device_hit, 0)
    )


def test_unified_tree_node_exposes_hash_chain():
    components = (ComponentType.FULL,)
    root = UnifiedTreeNode(components)
    root.hash_value = []
    parent = UnifiedTreeNode(components)
    parent.parent = root
    parent.hash_value = ["a", "b"]
    child = UnifiedTreeNode(components)
    child.parent = parent
    child.hash_value = ["c"]

    assert root.get_last_hash_value() is None
    assert child.get_last_hash_value() == "c"
    assert child.get_prefix_hash_values(parent) == ["a", "b"]


def test_connector_reduction_uses_attention_groups(monkeypatch):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = object()
    cache.attn_tp_group = object()
    cache.tp_group = object()
    cache.tp_world_size = 2
    calls = []

    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: 2,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op, group: calls.append(group),
    )

    cache._all_reduce_attn_groups(torch.tensor([1]), torch.distributed.ReduceOp.MIN)
    assert calls == [cache.attn_cp_group, cache.attn_tp_group]


def test_connector_reduction_uses_tp_fallback(monkeypatch):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = object()
    cache.tp_world_size = 2
    calls = []

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op, group: calls.append(group),
    )

    cache._all_reduce_attn_groups(torch.tensor([1]), torch.distributed.ReduceOp.MIN)
    assert calls == [cache.tp_group]


def test_connector_rejects_pipeline_parallelism():
    mixin = UnifiedCacheConnectorMixin()
    mixin.tree_components = (ComponentType.FULL,)

    try:
        mixin.init_connector(SimpleNamespace(), SimpleNamespace(pp_size=2))
    except ValueError as error:
        assert "pipeline parallelism" in str(error)
    else:
        raise AssertionError("Expected pipeline parallel connector rejection.")


def test_sparse_multi_component_layer_ranges():
    k0 = torch.zeros((8, 3), dtype=torch.uint8)
    k2 = torch.zeros((8, 5), dtype=torch.uint8)
    v0 = torch.zeros((8, 7), dtype=torch.uint8)
    v2 = torch.zeros((8, 11), dtype=torch.uint8)
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[k0, k2], [v0, v2]],
        layer_mapping={0: 0, 2: 1},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert len(pointers) == 8
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert len(pointers) == 4
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_lookup_returns_sparse_mamba_boundaries():
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.sources = {
        PoolName.KV: PoolName.KV,
        PoolName.MAMBA: PoolName.MAMBA,
    }
    identity_pool = SimpleNamespace(translate_indices=lambda indices: indices)
    connector.pools = {
        PoolName.KV: identity_pool,
        PoolName.MAMBA: identity_pool,
    }
    connector.stats = {"lookup": 0}
    connector._page_exists = lambda keys, transfer: (
        [True, True, True, True]
        if transfer.name == PoolName.KV
        else [False, True, False, True]
    )

    valid = connector.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["d"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )
    assert valid == [2, 4]


def test_partial_load_can_expand_a_side_pool_without_kv():
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.sources = {
        PoolName.KV: PoolName.KV,
        PoolName.SWA: PoolName.SWA,
    }
    identity_pool = SimpleNamespace(translate_indices=lambda indices: indices)
    connector.pools = {
        PoolName.KV: identity_pool,
        PoolName.SWA: identity_pool,
    }
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["page"],
        device_indices=torch.tensor([20, 21]),
    )

    expanded = connector._expand([swa], allow_partial=True, allow_missing_kv=True)

    assert len(expanded) == 1
    assert expanded[0].name == PoolName.SWA
    assert expanded[0].keys == ["page"]
    assert expanded[0].host_indices.tolist() == [20, 21]


def test_layerwise_load_reports_session_end_failure():
    class _Store:
        def batch_get_session_start(self, keys):
            return [0] * len(keys)

        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            return [sum(item) for item in sizes]

        def batch_get_session_end(self, keys):
            return [1] * len(keys)

    pool = SimpleNamespace(
        prepare_locations=lambda indices: [0],
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[1]],
            [[4]],
            [[0]],
        ),
    )
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.num_layers = 1
    connector.pools = {PoolName.KV: pool}
    connector.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    connector.layer_done_counter = LayerWiseLoadCounter(1)
    counter_index = connector.layer_done_counter.update_producer()

    connector._run_layer_wise_batch(
        counter_index,
        [
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page"],
                    host_indices=torch.tensor([0]),
                )
            ]
        ],
    )

    future = connector.layer_done_counter.futures[counter_index][0]
    assert future.done()
    assert isinstance(future.exception(), RuntimeError)


def test_load_waits_for_scheduler_stream(monkeypatch):
    event_calls = []
    loaded = threading.Event()

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_tree_connector.device_module, "Event", _Event)

    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.pending_loads = {"rid": [object()]}
    connector.layer_done_counter = SimpleNamespace(update_producer=lambda: 7)
    connector.load_queue = Queue()
    connector.stats = {"load": 0}

    def run_layer_wise(counter_index, transfers):
        assert event_calls == ["record", "synchronize"]
        assert counter_index == 7
        assert len(transfers) == 1
        loaded.set()

    connector._run_layer_wise_batch = run_layer_wise
    thread = threading.Thread(target=connector.load_thread_func, daemon=True)
    thread.start()

    assert connector.start_layer_wise_loading() == 7
    assert loaded.wait(timeout=5)
    connector.load_queue.join()
    connector.load_queue.put(None)
    thread.join(timeout=5)


def test_offload_runs_on_background_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []
    event_calls = []

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_tree_connector.device_module, "Event", _Event)

    class _Storage:
        def batch_set_v2(self, transfers):
            worker_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        translate_indices=lambda indices: indices,
        get_hybrid_pool_buffer=lambda: [],
    )
    connector = MooncakeTreeConnector.__new__(MooncakeTreeConnector)
    connector.page_size = 2
    connector.sources = {PoolName.KV: PoolName.KV}
    connector.pools = {PoolName.KV: pool}
    connector.storage = _Storage()
    connector.stats = {"lookup": 0, "load": 0, "offload": 0}
    connector.offload_queue = Queue()
    connector.offload_results = Queue()
    connector.offload_thread = threading.Thread(
        target=connector.offload_thread_func, daemon=True
    )
    connector.offload_thread.start()

    assert connector.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert started.wait(timeout=5)
    assert connector.num_completed_offloads() == 0
    assert worker_threads == [connector.offload_thread.ident]
    assert worker_threads[0] != caller_thread
    assert event_calls == ["record", "synchronize"]

    release.set()
    connector.offload_queue.join()
    assert connector.num_completed_offloads() == 1
    assert connector.pop_completed_offload()
    connector.offload_queue.put(None)
    connector.offload_thread.join(timeout=5)


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_connector_transfer(self, phase, node=None):
            assert phase == ConnectorTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    results = []
    connector = SimpleNamespace(
        offload=lambda transfers: True,
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    mixin = UnifiedCacheConnectorMixin()
    mixin.connector = connector
    mixin._components_tuple = (_Component(),)
    mixin.connector_offloads = []
    mixin._all_reduce_attn_groups = lambda tensor, op: None
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    mixin.inc_lock_ref = inc_lock_ref
    mixin.dec_lock_ref = lambda node, params: unlocks.append((node, params))
    node = SimpleNamespace(connector_offloaded=False)

    mixin.offload_connector_node(node)
    assert locks == [node]
    assert node.connector_offloaded
    assert not unlocks

    results.append(False)
    mixin.drain_connector_offloads()
    assert not node.connector_offloaded
    assert unlocks == [(node, lock_params)]


def test_async_offload_drains_only_common_tp_prefix():
    results = [True, True, True]
    connector = SimpleNamespace(
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    mixin = UnifiedCacheConnectorMixin()
    mixin.connector = connector
    nodes = [SimpleNamespace(connector_offloaded=True) for _ in range(3)]
    lock_params = [object() for _ in range(3)]
    mixin.connector_offloads = list(zip(nodes, lock_params))
    unlocks = []
    mixin.dec_lock_ref = lambda node, params: unlocks.append((node, params))

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            assert value.tolist() == [3]
            value.fill_(1)
        else:
            assert value.tolist() == [1]
            value.fill_(0)

    mixin._all_reduce_attn_groups = reduce_to_common_state
    mixin.drain_connector_offloads()

    assert reduce_calls == 2
    assert results == [True, True]
    assert mixin.connector_offloads == list(zip(nodes[1:], lock_params[1:]))
    assert not nodes[0].connector_offloaded
    assert nodes[1].connector_offloaded
    assert nodes[2].connector_offloaded
    assert unlocks == [(nodes[0], lock_params[0])]


def test_release_connector_request_cancels_queued_load():
    cancelled = []
    mixin = UnifiedCacheConnectorMixin()
    mixin.connector = SimpleNamespace(
        cancel_queued_load=lambda rid: cancelled.append(rid)
    )
    mixin._connector_markers = {"rid": object()}

    mixin.release_connector_request("rid")

    assert "rid" not in mixin._connector_markers
    assert cancelled == ["rid"]


def test_deepseek_v4_device_pool_group_maps_sparse_sidecars():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )

    def state_pool():
        return SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
        )

    kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kvcache.start_layer = 0
    kvcache.end_layer = 3
    kvcache.swa_page_size = 2
    kvcache.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[
            torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
        ]
    )
    kvcache.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
    )
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    group = resolve_hybrid_device_pool_group(kvcache, 2, None)
    assert group.num_layers == 3
    assert set(group.entry_map) == {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }
    assert group.sources[PoolName.DEEPSEEK_V4_C4] == PoolName.KV
    assert group.sources[PoolName.DEEPSEEK_V4_C4_STATE] == PoolName.SWA
    c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
    pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 2
    assert sizes == [5, 5]
    _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
    assert sizes == [[5]]
    assert offsets == [[5]]
    assert c4_pool.get_prepared_layer_range_meta([0], 1) is None


def test_qwen35_device_pool_group_maps_full_and_mamba_layers():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
    kvcache.use_mla = False
    kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1}
    kvcache.full_kv_pool = SimpleNamespace(
        size=6,
        k_scale_buffer=None,
        k_buffer=[torch.zeros((8, 3)), torch.zeros((8, 5))],
        v_buffer=[torch.zeros((8, 7)), torch.zeros((8, 11))],
    )
    req_pool = SimpleNamespace(
        mamba_ckpt_pool=None,
        mamba_map={1: 0, 3: 1},
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 5, 2, 3)),
                conv=[torch.zeros((2, 5, 4))],
            )
        ),
        translate_mamba_indices=lambda indices: indices,
    )

    group = resolve_hybrid_device_pool_group(kvcache, 2, req_pool)
    pools = group.entry_map
    assert group.num_layers == 4
    assert set(pools) == {PoolName.KV, PoolName.MAMBA}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.MAMBA: PoolName.MAMBA,
    }
    assert pools[PoolName.MAMBA].translate_indices(torch.tensor([1])).tolist() == [1]
    assert pools[PoolName.KV].get_prepared_layer_range_meta([0], 1) is None
    assert pools[PoolName.MAMBA].get_prepared_layer_range_meta([0], 0) is None
    pointers, sizes = pools[PoolName.KV].get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 4
    assert sizes == [24, 40, 56, 88]


def test_swa_connector_finish_maps_or_releases_slots():
    swa_allocator = _Allocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=64,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 128
    req = SimpleNamespace(swa_evicted_seqlen=0)
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.finish_connector_load(req, full, swa, prefix_len=256, success=True)
    mapped_full, mapped_swa = swa_allocator.mapping[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.swa_evicted_seqlen == 128

    component.finish_connector_load(req, full, swa, prefix_len=256, success=False)
    assert swa_allocator.freed[0].tolist() == [20, 21]


def test_mamba_connector_load_allocates_cache_and_request_slots():
    allocator = _Allocator(slots=torch.tensor([7, 8]))
    req_pool = SimpleNamespace(mamba_pool=allocator)
    component = MambaComponent.__new__(MambaComponent)
    component.cache = SimpleNamespace(
        req_to_token_pool=req_pool,
        evict=lambda params: None,
    )

    transfer = component.build_connector_transfer(
        phase=ConnectorTransferPhase.LOAD,
        keys=["a", "b"],
    )
    assert transfer.keys == ["b", "b"]
    assert transfer.device_indices.tolist() == [7, 8]

    req = SimpleNamespace(mamba_pool_idx=None)
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1]))
    component.finish_connector_load(req, full, transfer, prefix_len=2, success=True)
    assert req.mamba_pool_idx.item() == 8

    failed = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([9, 10]))
    component.finish_connector_load(req, full, failed, prefix_len=2, success=False)
    assert allocator.freed[-1].tolist() == [9, 10]


def test_overlapping_load_only_requeues_adopted_pages():
    mixin = UnifiedCacheConnectorMixin()
    mixin.page_size = 2
    mixin.token_to_kv_pool_allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda indices: indices + 1000
    )
    queued = {"second": ["stale"]}

    def load(rid, transfers):
        queued[rid] = list(transfers)
        return True

    mixin.connector = SimpleNamespace(
        cancel_queued_load=lambda rid: queued.pop(rid),
        load=load,
    )

    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b"],
        device_indices=torch.tensor([100, 101, 102, 103]),
    )
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["b"],
        device_indices=torch.tensor([200, 201]),
    )
    mamba = PoolTransfer(
        name=PoolName.MAMBA,
        keys=["b", "b"],
        device_indices=torch.tensor([300, 301]),
    )
    canonical_full = torch.tensor([10, 11, 12, 13])
    loaded = SimpleNamespace(
        device_indices=torch.cat([torch.tensor([1, 2]), canonical_full]),
        last_device_node=SimpleNamespace(
            component_data={
                ComponentType.MAMBA: SimpleNamespace(value=torch.tensor([30]))
            }
        ),
    )

    returned = mixin._retarget_connector_load(
        "second",
        [
            (SimpleNamespace(component_type=ComponentType.FULL), full),
            (SimpleNamespace(component_type=ComponentType.SWA), swa),
            (SimpleNamespace(component_type=ComponentType.MAMBA), mamba),
        ],
        loaded,
        device_hit_len=2,
        prefix_len=6,
        insert_result=InsertResult(
            prefix_len=4,
            mamba_exist=True,
            adopted_ranges={
                ComponentType.FULL: [(4, 6)],
                ComponentType.SWA: [(4, 6)],
            },
        ),
    )

    assert returned.tolist() == canonical_full.tolist()
    assert full.keys == ["b"]
    assert full.device_indices.tolist() == [12, 13]
    assert swa.device_indices.tolist() == [1012, 1013]
    assert mamba.device_indices.tolist() == [301]
    assert queued["second"] == [full, swa, mamba]


def test_select_adopted_pages_preserves_disjoint_page_ranges():
    mixin = UnifiedCacheConnectorMixin()
    mixin.page_size = 2
    indices = torch.arange(8)

    selected, keys = mixin._select_adopted_pages(
        indices,
        [(2, 4), (6, 8)],
        prefix_len=8,
        keys=["a", "b", "c", "d"],
    )

    assert selected.tolist() == [2, 3, 6, 7]
    assert keys == ["b", "d"]


def test_fully_overlapping_load_is_cancelled_without_requeue():
    mixin = UnifiedCacheConnectorMixin()
    mixin.page_size = 2
    calls = []
    mixin.connector = SimpleNamespace(
        cancel_queued_load=lambda rid: calls.append(("cancel", rid)),
        load=lambda rid, transfers: calls.append(("load", rid)) or True,
    )
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b"],
        device_indices=torch.tensor([100, 101, 102, 103]),
    )
    loaded = SimpleNamespace(
        device_indices=torch.tensor([10, 11, 12, 13]),
        last_device_node=object(),
    )

    returned = mixin._retarget_connector_load(
        "overlap",
        [(SimpleNamespace(component_type=ComponentType.FULL), full)],
        loaded,
        device_hit_len=0,
        prefix_len=4,
        insert_result=InsertResult(prefix_len=4, adopted_ranges={}),
    )

    assert returned.tolist() == [10, 11, 12, 13]
    assert calls == [("cancel", "overlap")]
