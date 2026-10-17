"""Supported plugin hooks; complete DiT events and post-save rank observations.

ARM initializes hooks and synchronizes before generate. At the native worker
return, a threading.Event wakes an observer. Its independent CPU process group
waits for every rank's save completion, without blocking the native response.
Only after the client stops E2E does COLLECT query CUDA events and write records.
"""

import inspect
import os
import socket
import threading
import time
import traceback
from contextlib import ExitStack, suppress
from datetime import timedelta
from pathlib import Path

from common import read_json, write_json
from control import receive, send
from runtime import inspect_worker, verify_attention
from timing import DiTEvents


class Observer:
    def __init__(self, worker):
        import torch
        import torch.distributed as dist

        self.worker = worker
        self.output = Path(os.environ["SGLANG_BENCH_OUTPUT"])
        self.plan = read_json(os.environ["SGLANG_BENCH_PLAN"])
        self.profile = self.plan["models"][os.environ["SGLANG_BENCH_MODEL"]]["profile"]
        self.model, self.info = inspect_worker(worker, self.profile)
        self.vsa_audit = None
        if self.profile["adapter"] == "fastwan21_14b":
            from fastwan import VSAExecutionAudit

            self.vsa_audit = VSAExecutionAudit(self.model, self.profile)
        # Dedicated CPU group: never races with scheduler collectives on its group.
        self.group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
        self.timer = (
            DiTEvents(torch, self.profile["sampling"]["num_inference_steps"])
            if "dit" in self.plan["modes"]
            else None
        )
        self.done = threading.Event()
        self.armed = False
        self.finished_at = None
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.bind(
            str(Path(os.environ["SGLANG_BENCH_CONTROL"]) / f"{worker.rank}.sock")
        )
        self.socket.listen(1)
        write_json(self.output / f"effective_rank{worker.rank}.json", self.info)
        threading.Thread(
            target=self.serve, daemon=True, name="benchmark-observer"
        ).start()

    def serve(self):
        import torch
        import torch.distributed as dist

        self.vsa_audit = getattr(self, "vsa_audit", None)
        torch.cuda.set_device(self.worker.local_rank)
        stream = None
        resources = ExitStack()
        try:
            connection, _ = self.socket.accept()
            connection.settimeout(7200)
            resources.enter_context(connection)
            stream = resources.enter_context(connection.makefile("rwb"))
            while True:
                command = receive(stream)
                if command["command"] == "close":
                    return
                if command["command"] != "arm":
                    raise ValueError("Expected arm")
                mode = command["mode"]
                self.phase = command.get("phase", "repeat")
                self.done.clear()
                if self.vsa_audit:
                    self.vsa_audit.rows.clear()
                if mode == "dit":
                    self.timer.attach(self.model)
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                torch.cuda.reset_peak_memory_stats()
                self.armed = True
                send(stream, {"ready": True})
                if not self.done.wait(timeout=7200):
                    raise TimeoutError("Native worker did not complete the request")
                # DISTRIBUTED E2E END: all workers finished native forward + save.
                # Independent CPU observer; no wait in the generation thread.
                dist.barrier(group=self.group)
                barrier_at = time.perf_counter()
                command = receive(stream)  # client sends only AFTER response E2E
                if command["command"] != "collect":
                    raise ValueError("Expected collect")
                if mode == "dit":
                    self.timer.detach()
                torch.cuda.synchronize()  # OUTSIDE both E2E intervals
                self.info["attention_implementations"] = verify_attention(self.model)
                write_json(
                    self.output / f"effective_rank{self.worker.rank}.json",
                    self.info,
                )
                local = dict(
                    rank=self.worker.rank,
                    response_e2e_s=command["response_e2e_s"]
                    if self.worker.rank == 0
                    else None,
                    worker_finished_s=self.finished_at - command["started"],
                    distributed_e2e_s=barrier_at - command["started"],
                    dit_step_ms=self.timer.elapsed_ms() if mode == "dit" else None,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                )
                if self.vsa_audit:
                    local["vsa_execution"] = self.vsa_audit.collect()
                write_json(
                    self.output / f"{command['tag']}.rank{self.worker.rank}.json",
                    local,
                )
                self.armed = False
                send(stream, local)
        except BaseException:
            detail = traceback.format_exc()
            error_path = self.output / f"error_rank{self.worker.rank}.json"
            if not error_path.exists():
                write_json(
                    error_path,
                    dict(phase="observer", status="failed", traceback=detail),
                )
            if stream is not None and not stream.closed:
                try:
                    send(stream, {"error": detail})
                except (OSError, ValueError):
                    pass
        finally:
            try:
                if self.timer:
                    self.timer.detach()
                if self.vsa_audit:
                    self.vsa_audit.handle.remove()
            finally:
                with suppress(OSError, ValueError):
                    resources.close()
                self.socket.close()


def install_hooks():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    @plugin_hook(
        "sglang.multimodal_gen.runtime.managers.gpu_worker.GPUWorker.__init__",
        HookType.AROUND,
    )
    def initialize(original, worker, *args, **kwargs):
        rank = 0
        try:
            if os.environ.get("SGLANG_BENCH_CONTROL"):
                import torch
                from gpu_wait import check_device_identity

                bound = inspect.signature(original).bind(worker, *args, **kwargs)
                rank = bound.arguments["rank"]
                selected = read_json(os.environ["SGLANG_BENCH_PLAN"])[
                    "selected_gpu_uuids"
                ]
                check_device_identity(
                    torch, bound.arguments["local_rank"], selected[rank]
                )
            result = original(worker, *args, **kwargs)
            if os.environ.get("SGLANG_BENCH_CONTROL"):
                worker._benchmark_observer = Observer(worker)
            return result
        except BaseException as error:
            if os.environ.get("SGLANG_BENCH_OUTPUT"):
                import torch

                write_json(
                    Path(os.environ["SGLANG_BENCH_OUTPUT"]) / f"error_rank{rank}.json",
                    dict(
                        phase="load",
                        status="memory_insufficient"
                        if isinstance(error, torch.OutOfMemoryError)
                        else "failed",
                        traceback=traceback.format_exc(),
                    ),
                )
            raise

    @plugin_hook(
        "sglang.multimodal_gen.runtime.managers.gpu_worker.GPUWorker._execute_forward_common",
        HookType.AROUND,
    )
    def observe(original, worker, req, *args, **kwargs):
        observer = getattr(worker, "_benchmark_observer", None)
        if observer is not None:
            import torch

            forward = kwargs["forward_fn"]

            def measured_forward():
                try:
                    return forward()
                except torch.OutOfMemoryError:
                    write_json(
                        observer.output / f"error_rank{worker.rank}.json",
                        dict(
                            phase=observer.phase,
                            status="memory_insufficient"
                            if observer.phase == "warmup"
                            else "failed",
                            traceback=traceback.format_exc(),
                        ),
                    )
                    raise

            kwargs["forward_fn"] = measured_forward
        try:
            result = original(worker, req, *args, **kwargs)
            if (
                observer is not None
                and getattr(result, "error", None)
                and not (observer.output / f"error_rank{worker.rank}.json").exists()
            ):
                write_json(
                    observer.output / f"error_rank{worker.rank}.json",
                    dict(phase="generation", status="failed", traceback=result.error),
                )
            return result
        finally:
            if observer is not None and observer.armed:
                observer.finished_at = time.perf_counter()
                observer.done.set()  # wake CPU observer; never wait here
