"""Native SGLang configuration; no policy-alignment monkey patches."""

import os
from dataclasses import asdict
from pathlib import Path

from common import ROOT


def launch_environment(env, scripts):
    env = {
        k: v
        for k, v in env.items()
        if not k.startswith(("SGLANG_", "MINIMAX_H3_", "FASTVIDEO_"))
    }
    env.update(
        SGLANG_BENCH_SOURCE_ROOT=str(ROOT),
        SGLANG_PLUGINS="sglang_bench",
        SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_bench",
        SGLANG_DIFFUSION_STAGE_LOGGING="0",
        PYTHONPATH=os.pathsep.join([str(scripts), str(ROOT / "python")]),
    )
    compat = os.environ.get("BENCH_CUDA_COMPAT")
    if compat:
        if not (Path(compat) / "libcuda.so.1").is_file():
            raise ValueError("BENCH_CUDA_COMPAT must contain libcuda.so.1")
        env["LD_LIBRARY_PATH"] = compat + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    media = ROOT / ".cache/benchmark/tools/bin"
    if media.exists():
        env["PATH"] = str(media) + os.pathsep + env["PATH"]
        env["IMAGEIO_FFMPEG_EXE"] = str(media / "ffmpeg")
    return env


def server_kwargs(p, path, gpus):
    if p["adapter"] == "fastwan21_14b":
        from sglang.multimodal_gen.configs.pipeline_configs.wan import (
            FastWan2_1_T2V_480P_Config,
        )

        return dict(
            model_path=str(path),
            model_id=p["model_id"].rsplit("/", 1)[-1],
            num_gpus=gpus,
            pipeline_config=FastWan2_1_T2V_480P_Config(),
            pipeline_class_name="WanDMDPipeline",
            sp_degree=gpus,
            ulysses_degree=gpus,
            attention_backend="video_sparse_attn",
            attention_backend_config={"VSA_sparsity": p["vsa_sparsity"]},
            warmup_mode="off",
            **p["server"],
        )
    from sglang_bench.v1 import FastH3V1PipelineConfig

    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        FastH3PipelineConfig,
    )

    config = (
        FastH3V1PipelineConfig() if p["variant"] == "v1" else FastH3PipelineConfig()
    )
    return dict(
        model_path=str(path),
        model_id=p["model_id"].rsplit("/", 1)[-1],
        num_gpus=gpus,
        pipeline_config=config,
        pipeline_class_name="FastH3V1BaselinePipeline"
        if p["variant"] == "v1"
        else "FastH3Pipeline",
        sp_degree=gpus,
        ulysses_degree=gpus,
        component_attention_backends={"transformer": "video_sparse_attn_h3"},
        attention_backend_config={"VSA_sparsity": p["vsa_sparsity"]},
        warmup_mode="off",
        **p["server"],
    )


def make_request(p, output):
    s = p["sampling"]
    if p["adapter"] == "fastwan21_14b":
        return dict(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            **{
                k: s[k]
                for k in [
                    "height",
                    "width",
                    "num_frames",
                    "fps",
                    "seed",
                    "guidance_scale",
                    "num_inference_steps",
                ]
            },
            flow_shift=p["contract"]["flow_shift"],
            save_output=True,
            return_file_paths_only=True,
            output_path=str(output.parent),
            output_file_name=output.name,
        )
    return dict(
        prompt=p["prompt"],
        seed=s["seed"],
        task=s["task"],
        conditions=[],
        target=s["target"],
        num_inference_steps=s["num_inference_steps"],
        flow_shift=p["contract"]["video_shift"],
        audio_flow_shift=p["contract"]["audio_shift"],
        quality=s["quality"],
        save_output=True,
        return_file_paths_only=True,
        output_path=str(output.parent),
        output_file_name=output.name,
    )


def inspect_worker(worker, profile):
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import inspect_worker as inspect_wan

        return inspect_wan(worker, profile)
    import torch

    model = worker.pipeline.get_module("transformer")
    blocks = model.blocks
    backends = [str(block.attn._attention_backend_enum) for block in blocks]
    if not all(
        getattr(block.attn, "to_gate_compress", None) is not None for block in blocks
    ):
        raise ValueError("Missing trained VSA compression gates")
    dtype = str(model.blocks[0].attn.qkv_proj.weight.dtype)
    if dtype != "torch.bfloat16":
        raise ValueError(f"Expected BF16 DiT attention weights, got {dtype}")
    metadata = worker.pipeline.release_metadata
    if list(metadata.dmd_denoising_steps) != profile["contract"]["dmd_denoising_steps"]:
        raise ValueError("Loaded pipeline DMD schedule mismatch")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    info = dict(
        dit_attention_dtype=dtype,
        dit_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rank=worker.rank,
        name=props.name,
        uuid=str(props.uuid),
        capability=[props.major, props.minor],
        total_memory_gib=props.total_memory / 2**30,
        attention_implementations=sorted(set(backends)),
        trained_gate_count=len(blocks),
        pipeline_config=asdict(worker.server_args.pipeline_config),
        server_args=asdict(worker.server_args),
        cpu_threads=torch.get_num_threads(),
    )
    return model, info


def video_metadata(path, require_audio=True):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        video = container.streams.video[0]
        result = dict(
            width=video.width,
            height=video.height,
            fps=float(video.average_rate),
            frames=sum(1 for _ in container.decode(video=0)),
        )
    with av.open(str(path)) as container:
        if not container.streams.audio:
            if require_audio:
                raise ValueError("Missing audio stream")
            return result
        audio = list(container.decode(audio=0))
        if not audio or not all(
            np.isfinite(frame.to_ndarray()).all() for frame in audio
        ):
            raise ValueError("Missing or non-finite audio")
    return result


def verify_attention(model):
    """Native H3 binds attention lazily; verify actual backends AFTER warmup."""
    from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

    if not model.blocks:
        raise ValueError("Unexpected executed attention backends: no blocks")
    wan = type(model.blocks[0]).__name__ == "WanTransformerBlock_VSA"
    if wan:
        implementations = {
            type(block.attn1.attn_impl).__name__ for block in model.blocks
        }
        if implementations != {"VideoSparseAttentionImpl"}:
            raise ValueError(
                f"Unexpected executed attention backends: {implementations}"
            )
        return sorted(implementations)
    actual = {block.attn._attention_backend_enum for block in model.blocks}
    backends = sorted(str(backend) for backend in actual)
    expected = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    if actual != {expected}:
        raise ValueError(f"Unexpected executed attention backends: {backends}")
    return backends


def execution_policy(server):
    """Use native residency selectors; legacy boolean flags alone are ambiguous."""
    keys = [
        "performance_mode",
        "cpu_offload_components",
        "layerwise_offload_components",
        "pin_cpu_memory",
        "use_fsdp_inference",
        "encoder_parallel",
        "disable_conditioning_cache",
        "enable_torch_compile",
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "enable_cfg_parallel",
        "async_output_save",
    ]
    policy = {key: getattr(server, key) for key in keys}
    policy["component_residency"] = {
        name: server.residency_mode(name)
        for name in ["transformer", "text_encoder", "video_vae", "audio_vae"]
    }
    return policy
