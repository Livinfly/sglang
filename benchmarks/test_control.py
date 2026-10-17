"""Exercise two-rank observer ordering without CUDA or model weights."""

import copy
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common import HERE, read_json, write_json
from config import validate_profile
from control import close_clients, connect, exchange, send
from runtime import verify_attention
from sglang_bench.observer import Observer
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


class ControlTests(unittest.TestCase):
    def test_native_backend_enum_is_accepted_but_dense_and_unbound_are_rejected(self):
        def model(*backends):
            return SimpleNamespace(blocks=[
                SimpleNamespace(attn=SimpleNamespace(_attention_backend_enum=backend))
                for backend in backends
            ])

        sparse = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
        self.assertEqual(str(sparse), "video_sparse_attn_h3")
        self.assertEqual(verify_attention(model(sparse)), ["video_sparse_attn_h3"])
        for backends in [(), (None,), (AttentionBackendEnum.TORCH_SDPA,),
                         (sparse, AttentionBackendEnum.TORCH_SDPA), (str(sparse),)]:
            with self.assertRaises(ValueError):
                verify_attention(model(*backends))

    def test_broken_stream_close_still_closes_every_socket(self):
        clients = [(Mock(), Mock()), (Mock(), Mock())]
        clients[0][1].write.side_effect = BrokenPipeError("observer exited")
        clients[0][1].close.side_effect = BrokenPipeError("flush failed")
        close_clients(clients)
        for sock, stream in clients:
            stream.close.assert_called_once()
            sock.close.assert_called_once()

    def test_observer_sends_original_error_before_closing_stream(self):
        fake_dist = SimpleNamespace(barrier=lambda **kw: None)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda n: None, synchronize=lambda: None),
            distributed=fake_dist,
        )
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            sys.modules, {"torch": fake_torch, "torch.distributed": fake_dist}
        ):
            directory = Path(folder)
            observer = Observer.__new__(Observer)
            observer.worker = SimpleNamespace(rank=0, local_rank=0)
            observer.output = directory
            observer.model = object()
            observer.info = {}
            observer.timer = Mock()
            observer.group = object()
            observer.socket = socket.socket(socket.AF_UNIX)
            observer.socket.bind(str(directory / "0.sock"))
            observer.socket.listen(1)
            # Reach COLLECT through the real socket protocol, then fail validation.
            observer.done = Mock()
            observer.done.wait.return_value = True
            fake_torch.cuda.reset_peak_memory_stats = lambda: None
            with patch("sglang_bench.observer.verify_attention", side_effect=ValueError("backend mismatch")):
                thread = threading.Thread(target=observer.serve)
                thread.start()
                clients = connect(directory, 1)
                try:
                    exchange(clients, dict(command="arm", mode="dit", phase="warmup"))
                    with self.assertRaisesRegex(RuntimeError, "ValueError: backend mismatch"):
                        exchange(clients, dict(command="collect"))
                finally:
                    close_clients(clients)
                    thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIn("ValueError: backend mismatch", read_json(directory / "error_rank0.json")["traceback"])

    def test_installed_plugin_is_inert_without_explicit_opt_in(self):
        import sglang_bench

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sglang_bench, "_registered", False),
        ):
            sglang_bench.register()
            self.assertFalse(sglang_bench._registered)

    def test_gpu_identity_failure_prevents_model_loading(self):
        from sglang_bench.observer import install_hooks

        hooks = {}

        def register(target, kind):
            def wrap(function):
                hooks[target.rsplit(".", 1)[-1]] = function
                return function

            return wrap

        plugin = SimpleNamespace(
            HookType=SimpleNamespace(AROUND="around"), plugin_hook=register
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.multimodal_gen.runtime.platforms.plugins": plugin,
                "torch": SimpleNamespace(OutOfMemoryError=MemoryError),
            },
        ):
            install_hooks()
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                write_json(
                    directory / "plan.json", {"selected_gpu_uuids": ["GPU-a", "GPU-b"]}
                )

                def native_init(worker, local_rank, rank, master_port, server_args):
                    raise AssertionError(
                        "Model loading must not start on the wrong GPU"
                    )

                with (
                    patch.dict(
                        os.environ,
                        {
                            "SGLANG_BENCH_CONTROL": folder,
                            "SGLANG_BENCH_OUTPUT": folder,
                            "SGLANG_BENCH_PLAN": str(directory / "plan.json"),
                        },
                    ),
                    patch(
                        "gpu_wait.check_device_identity",
                        side_effect=ValueError("UUID mismatch"),
                    ) as check,
                ):
                    with self.assertRaisesRegex(ValueError, "UUID mismatch"):
                        hooks["__init__"](
                            native_init,
                            object(),
                            local_rank=1,
                            rank=1,
                            master_port=1234,
                            server_args=None,
                        )
                    self.assertEqual(check.call_args.args[1:], (1, "GPU-b"))
                    self.assertEqual(
                        read_json(directory / "error_rank1.json")["phase"], "load"
                    )

    def test_profile_rejects_wrong_steps_and_async_save(self):
        for variant, count in [("v1", 4), ("v2", 8)]:
            p = read_json(HERE / f"configs/fasth3_{variant}.json")
            validate_profile(p)
            self.assertEqual(p["sampling"]["num_inference_steps"], count)
            bad = copy.deepcopy(p)
            bad["sampling"]["num_inference_steps"] += 1
            with self.assertRaises(ValueError):
                validate_profile(bad)
            bad = copy.deepcopy(p)
            bad["server"]["async_output_save"] = True
            with self.assertRaises(ValueError):
                validate_profile(bad)

    def test_post_save_barrier_does_not_wait_for_client_collect(self):
        barrier = threading.Barrier(2, timeout=5)
        syncs = []
        fake_dist = SimpleNamespace(barrier=lambda **kw: barrier.wait())
        fake_cuda = SimpleNamespace(
            set_device=lambda n: None,
            synchronize=lambda: syncs.append(time.perf_counter()),
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 10,
            max_memory_reserved=lambda: 20,
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda, distributed=fake_dist)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                sys.modules, {"torch": fake_torch, "torch.distributed": fake_dist}
            ),
        ):
            directory = Path(folder)
            observers, threads = [], []
            for rank in range(2):
                observer = Observer.__new__(Observer)
                observer.worker = SimpleNamespace(rank=rank, local_rank=rank)
                observer.output = directory
                observer.model = SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            attn=SimpleNamespace(
                                _attention_backend_enum=AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
                            )
                        )
                    ]
                )
                observer.info = {}
                observer.timer = Mock()
                observer.timer.elapsed_ms.return_value = [2, 3]
                observer.group = object()
                observer.done = threading.Event()
                observer.armed = False
                observer.socket = socket.socket(socket.AF_UNIX)
                observer.socket.bind(str(directory / f"{rank}.sock"))
                observer.socket.listen(1)
                observers.append(observer)
                thread = threading.Thread(target=observer.serve)
                thread.start()
                threads.append(thread)
            clients = connect(directory, 2)
            try:
                for mode in ["dit", "e2e"]:
                    exchange(clients, {"command": "arm", "mode": mode})
                    before = len(syncs)
                    started = time.perf_counter()
                    for observer in observers:
                        observer.finished_at = time.perf_counter()
                        observer.done.set()
                    # Simulated media response transport: observers may finish before the client.
                    time.sleep(0.05)
                    self.assertEqual(len(syncs), before)  # no post-forward GPU sync
                    response = time.perf_counter() - started
                    records = exchange(
                        clients,
                        dict(
                            command="collect",
                            started=started,
                            response_e2e_s=response,
                            tag=mode,
                        ),
                    )
                    self.assertEqual([r["rank"] for r in records], [0, 1])
                    self.assertEqual(len(syncs), before + 2)
                    for row in records:
                        self.assertGreaterEqual(
                            row["distributed_e2e_s"], row["worker_finished_s"]
                        )
                        self.assertEqual(
                            row["dit_step_ms"], [2, 3] if mode == "dit" else None
                        )
                    self.assertTrue(
                        all(o.timer.attach.call_count == 1 for o in observers)
                    )
            finally:
                for sock, stream in clients:
                    send(stream, {"command": "close"})
                    stream.close()
                    sock.close()
                for thread in threads:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
            self.assertFalse(list(directory.glob("error_*.json")))


if __name__ == "__main__":
    unittest.main()
