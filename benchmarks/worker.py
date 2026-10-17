"""One native DiffGenerator client; SGLang owns its multi-GPU scheduler processes."""

import argparse
import importlib.metadata
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from common import SCHEMA, aggregate, read_json, schedule, sha256, write_json
from control import close_clients, connect, exchange
from runtime import execution_policy, make_request, server_kwargs, video_metadata


class NativeErrors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    profile = plan["models"][args.model]["profile"]
    generator, clients = None, []
    phase, records = "load", []
    control_dir = tempfile.TemporaryDirectory(prefix="sgbench-")
    os.environ.update(
        SGLANG_BENCH_PLAN=str(args.plan),
        SGLANG_BENCH_MODEL=args.model,
        SGLANG_BENCH_OUTPUT=str(args.output),
        SGLANG_BENCH_CONTROL=control_dir.name,
    )
    # Host-local compiler caches avoid network-filesystem stale handles, shared across groups.
    cache = Path(
        os.environ.get("BENCH_CACHE_DIR", f"/tmp/sglang-benchmark-{os.getuid()}")
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_DIFFUSION_CACHE_ROOT"] = str(cache)

    def dump(name, value):
        write_json(args.output / name, value)

    try:
        import torch

        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
            logger,
        )

        errors = NativeErrors()
        logger.addHandler(errors)
        dump("progress.json", dict(status="loading"))
        started = time.perf_counter()
        generator = DiffGenerator.from_pretrained(
            **server_kwargs(profile, plan["models"][args.model]["path"], args.gpus)
        )
        load_s = time.perf_counter() - started
        clients = connect(Path(control_dir.name), args.gpus)
        rank_info = [
            read_json(args.output / f"effective_rank{rank}.json")
            for rank in range(args.gpus)
        ]
        server = asdict(generator.server_args)
        effective = dict(
            server_args=server,
            pipeline_config=rank_info[0]["pipeline_config"],
            profile=profile,
            devices=rank_info,
            expected_dit_forwards=profile["sampling"]["num_inference_steps"],
            resolved_execution=execution_policy(generator.server_args),
            attention_implementations=rank_info[0]["attention_implementations"],
            python=sys.version,
            cuda=torch.version.cuda,
            load_seconds_excluded=load_s,
            packages={
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "transformers",
                    "diffusers",
                    "triton",
                    "av",
                    "sglang-kernel",
                    *(["vsa"] if profile["adapter"] == "fastwan21_14b" else []),
                ]
            },
            environment={
                k: v
                for k, v in os.environ.items()
                if k.startswith(("SGLANG_", "MINIMAX_H3_", "NCCL_"))
                or k
                in {
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "OMP_NUM_THREADS",
                    "LD_LIBRARY_PATH",
                }
            },
        )
        dump("effective_config.json", effective)
        for mode, warmup, repeat in schedule(
            plan["modes"], plan["warmup"], plan["repeats"]
        ):
            phase = "warmup" if warmup else "repeat"
            tag = f"{mode}_{phase}_{repeat}"
            output = args.output / f"{tag}.mp4"
            sample = make_request(profile, output)
            dump("progress.json", dict(status="running", tag=tag))
            exchange(clients, dict(command="arm", mode=mode, phase=phase))
            errors.messages.clear()
            # E2E START: native client request preparation + IPC + full generation + save.
            started = time.perf_counter()
            result = generator.generate(sample)
            response_s = time.perf_counter() - started  # E2E END: saved media returned.
            # No added CUDA synchronization inside client E2E in either mode.
            if isinstance(result, list):
                if len(result) != 1:
                    raise RuntimeError("Expected exactly one output")
                result = result[0]
            if result is None:
                raise RuntimeError(
                    "\n".join(errors.messages) or "Native API returned no result"
                )
            per_rank = exchange(
                clients,
                dict(
                    command="collect",
                    started=started,
                    response_e2e_s=response_s,
                    tag=tag,
                ),
            )
            effective["devices"] = [
                read_json(args.output / f"effective_rank{rank}.json")
                for rank in range(args.gpus)
            ]
            effective["attention_implementations"] = effective["devices"][0][
                "attention_implementations"
            ]
            dump("effective_config.json", effective)
            record = aggregate(
                per_rank, mode=mode, tag=tag, warmup=warmup, repeat=repeat
            )
            path = Path(result.output_file_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Native API did not save a video")
            media = video_metadata(
                path, require_audio=profile["adapter"] != "fastwan21_14b"
            )  # Always confirm actual shape and audio OUTSIDE timing.
            if any(
                media[k] != profile["sampling"][s]
                for k, s in [
                    ("width", "width"),
                    ("height", "height"),
                    ("fps", "fps"),
                    ("frames", "num_frames"),
                ]
            ):
                raise ValueError(f"Unexpected output media: {media}")
            record.update(
                video_file=str(path.relative_to(args.output)), video_metadata=media
            )
            if plan["validate_video"]:
                record["video_sha256"] = sha256(path)
            records.append(record)
            dump(
                "metrics.json",
                dict(status="running", timing_schema_version=SCHEMA, records=records),
            )
            print(
                f"{tag}: E2E={response_s:.3f}s, DiT={record.get('dit_forward_sum_max_rank_s')}",
                flush=True,
            )
        dump(
            "metrics.json",
            dict(status="complete", timing_schema_version=SCHEMA, records=records),
        )
        dump("progress.json", dict(status="complete"))
    except BaseException:
        # Do not label a generic native RuntimeError as OOM based on advisory log text.
        dump(
            "error_client.json",
            dict(phase=phase, status="failed", traceback=traceback.format_exc()),
        )
        raise
    finally:
        try:
            close_clients(clients)
        finally:
            try:
                if generator is not None:
                    generator.shutdown()
            finally:
                control_dir.cleanup()


if __name__ == "__main__":
    main()
