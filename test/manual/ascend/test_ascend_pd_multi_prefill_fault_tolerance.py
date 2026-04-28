"""Manual e2e test for the per-Prefill MemFabric store SPOF fix.

Goal: verify that with 2 Prefill + 2 Decode Ascend instances, killing **one**
Prefill does **not** affect the surviving P + the Decode side. Before the
SPOF fix every P/D shared a single ``ASCEND_MF_STORE_URL`` hosted by the
first Prefill, so killing P1 took everyone down.

This test is intentionally **manual / nightly only**:
  * It requires real NPU hardware (>= 4 NPUs across 2 hosts or simulated
    multi-process in one host with extra ports).
  * It launches multiple SGLang servers and a small router.
  * Run it with::

        python3 -m unittest test.manual.ascend.test_ascend_pd_multi_prefill_fault_tolerance

Prerequisites:
  * ``ASCEND_MF_STORE_URL`` MUST be **unset** (we are validating the new
    self-hosted-per-Prefill mode). If it's set the test will skip.
  * ``memfabric_hybrid`` is installed.
  * ``ASCEND_RT_VISIBLE_DEVICES`` selects which NPUs to use.

What this test does:
  1. Start P1, P2 (each on its own ``--disaggregation-bootstrap-port``; the
     MemFabric store port is auto-derived as ``bootstrap_port + 1``).
  2. Start D1, D2 sharing both Prefills via the mini-LB router.
  3. Issue a baseline request; confirm 200 OK.
  4. Kill P1.
  5. Issue more requests; the LB should route them to P2 + the still-alive
     decoders without errors. Before the fix this step would hang because
     every D was registered against P1's MemFabric store.

If you do not have a multi-host setup, you can still exercise the same code
paths on a single host by:
  * Setting different ``--port`` / ``--disaggregation-bootstrap-port`` per
    process (the store port follows ``bootstrap_port + 1`` automatically).
  * Pinning ``ASCEND_RT_VISIBLE_DEVICES`` per process so they don't fight for
    NPUs.
"""

import os
import time
import unittest
import warnings
from typing import List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_with_error_check,
)


def _wait_ready(url: str, timeout: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH) -> None:
    start = time.perf_counter()
    while True:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if time.perf_counter() - start > timeout:
            raise RuntimeError(f"Server {url} not ready in {timeout}s")
        time.sleep(1)


def _generate(lb_url: str, prompt: str, max_tokens: int = 16) -> requests.Response:
    return requests.post(
        f"{lb_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
        },
        timeout=60,
    )


@unittest.skipIf(
    os.getenv("ASCEND_MF_STORE_URL"),
    "ASCEND_MF_STORE_URL is set -- this test only validates the new "
    "self-hosted-per-Prefill mode",
)
@unittest.skipUnless(
    os.getenv("SGLANG_RUN_ASCEND_MULTI_P_FAULT_TEST") == "1",
    "Manual test. Set SGLANG_RUN_ASCEND_MULTI_P_FAULT_TEST=1 and ensure "
    "you have >= 4 NPUs available before running.",
)
class TestAscendMultiPrefillFaultTolerance(CustomTestCase):
    model = os.getenv("SGLANG_TEST_ASCEND_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    host = "127.0.0.1"
    base_port = 30000

    @classmethod
    def setUpClass(cls):
        cls.processes: List = []
        cls.p1_url = f"http://{cls.host}:{cls.base_port + 100}"
        cls.p2_url = f"http://{cls.host}:{cls.base_port + 110}"
        cls.d1_url = f"http://{cls.host}:{cls.base_port + 200}"
        cls.d2_url = f"http://{cls.host}:{cls.base_port + 210}"
        cls.lb_url = f"http://{cls.host}:{cls.base_port}"

        common_p = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--host",
            cls.host,
        ]
        common_d = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--host",
            cls.host,
        ]

        # MemFabric store port is auto-derived as bootstrap_port + 1, so we
        # only need to keep enough gap between the two Prefills' bootstrap
        # ports (>=2) to avoid colliding store sockets.
        cls.processes.append(
            popen_with_error_check(
                common_p
                + [
                    "--port",
                    str(cls.base_port + 100),
                    "--disaggregation-bootstrap-port",
                    str(cls.base_port + 101),
                ]
            )
        )
        cls.processes.append(
            popen_with_error_check(
                common_p
                + [
                    "--port",
                    str(cls.base_port + 110),
                    "--disaggregation-bootstrap-port",
                    str(cls.base_port + 111),
                ]
            )
        )
        cls.processes.append(
            popen_with_error_check(
                common_d
                + [
                    "--port",
                    str(cls.base_port + 200),
                ]
            )
        )
        cls.processes.append(
            popen_with_error_check(
                common_d
                + [
                    "--port",
                    str(cls.base_port + 210),
                ]
            )
        )

        for u in [cls.p1_url, cls.p2_url, cls.d1_url, cls.d2_url]:
            _wait_ready(u + "/health")

        cls.processes.append(
            popen_with_error_check(
                [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    cls.p1_url,
                    "--prefill",
                    cls.p2_url,
                    "--decode",
                    cls.d1_url,
                    "--decode",
                    cls.d2_url,
                    "--host",
                    cls.host,
                    "--port",
                    str(cls.base_port),
                ]
            )
        )
        _wait_ready(cls.lb_url + "/health")

    @classmethod
    def tearDownClass(cls):
        for p in cls.processes:
            try:
                kill_process_tree(p.pid)
            except Exception as e:
                warnings.warn(f"failed killing {p.pid}: {e}")
        time.sleep(5)

    def test_kill_first_prefill_does_not_break_others(self):
        # Baseline.
        r = _generate(self.lb_url, "Hello", max_tokens=8)
        self.assertEqual(r.status_code, 200, msg=r.text)

        # Kill P1 (the host that, in the legacy single-store world, owned
        # ASCEND_MF_STORE_URL and therefore took down the whole fleet when
        # killed).
        p1_proc = self.processes[0]
        kill_process_tree(p1_proc.pid)
        time.sleep(3)

        # Issue a few more requests. The mini-LB will retry on a healthy
        # Prefill (P2). With the SPOF fix, P2 owns its own MemFabric store
        # and the Decode side keeps an engine bound to P2 -- so this should
        # succeed without any global rendezvous failure.
        for prompt in ["Hi", "How are you", "Tell me a joke"]:
            r = _generate(self.lb_url, prompt, max_tokens=8)
            self.assertEqual(r.status_code, 200, msg=r.text)


if __name__ == "__main__":
    unittest.main()
