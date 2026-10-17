"""Resolve the real native model/request/ServerArgs without loading weights."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from common import ROOT, read_json, write_json
from config import model_manifest, validate_profile
from runtime import execution_policy, launch_environment, make_request, server_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("SGLANG_BENCH_PREVIEW_READY") != "1":
        env = launch_environment(dict(os.environ), Path(__file__).resolve().parent)
        env["SGLANG_BENCH_PREVIEW_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    import torch

    if max(args.gpus) > torch.cuda.device_count():
        parser.error("Preview GPU counts must fit the visible CUDA devices")
    from sglang_bench import register

    register()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        HookRegistry,
        apply_plugin_hooks,
    )

    apply_plugin_hooks()
    for method in ["__init__", "_execute_forward_common"]:
        target = "sglang.multimodal_gen.runtime.managers.gpu_worker.GPUWorker." + method
        if target not in HookRegistry._patched:
            raise RuntimeError("Measurement hook was not applied: " + target)
    from sglang.multimodal_gen.registry import get_model_info
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        minimax_h3_resolve_plan,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    p = read_json(args.config)
    validate_profile(p)
    manifest = model_manifest(args.model_path, p, args.gpus)
    info = get_model_info(
        str(args.model_path),
        backend="sglang",
        model_id=p["model_id"].rsplit("/", 1)[-1],
    )
    sample = make_request(p, ROOT / "preview.mp4")
    params = info.sampling_param_cls(**sample)
    if p["adapter"] == "fastwan21_14b":
        shape = {
            k: getattr(params, k) for k in ["width", "height", "num_frames", "fps"]
        }
        shape["frame_count"] = shape.pop("num_frames")
    else:
        resolved = minimax_h3_resolve_plan(
            params.build_request_extra()["minimax_h3_canonical_request"]
        )
        shape = resolved.shape
    for actual, expected in [
        ("width", "width"),
        ("height", "height"),
        ("frame_count", "num_frames"),
        ("fps", "fps"),
    ]:
        if shape[actual] != p["sampling"][expected]:
            raise ValueError(f"Native request {actual} does not match profile")
    rows = []
    for g in args.gpus:
        server = ServerArgs.from_kwargs(**server_kwargs(p, args.model_path, g))
        if (
            server.sp_degree != g
            or server.ulysses_degree != g
            or server.tp_size != 1
            or server.enable_cfg_parallel
        ):
            raise ValueError("Unexpected resolved parallelism")
        rows.append(
            dict(
                gpus=g,
                server_args=asdict(server),
                resolved_execution=execution_policy(server),
            )
        )
    result = dict(
        profile=p,
        pipeline_class=info.pipeline_cls.__name__,
        sampling_class=info.sampling_param_cls.__name__,
        resolved_shape=shape,
        trained_gate_count=manifest["trained_gate_count"],
        configurations=rows,
        scope="Native registry/request/ServerArgs validated; no weights loaded or inference performed",
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
