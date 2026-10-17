"""Pinned FastWan admission and native execution observation."""

import json
import math
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common import ROOT, read_json, sha256

MODEL_ID = "FastVideo/FastWan2.1-T2V-14B-Diffusers"
REVISION = "ab7d6149c10bb41ba39226e2262cd2e9ee85b88e"


def validate_profile(p):
    if p["model_id"] != MODEL_ID:
        raise ValueError("Require the selected FastWan 2.1 14B release")
    s = p["sampling"]
    for key in ["height", "width", "num_frames", "fps"]:
        if type(s[key]) is not int or s[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if s["height"] % 16 or s["width"] % 16 or s["num_frames"] % 4 != 1:
        raise ValueError("Invalid Wan VAE/patch shape")
    if (s["num_inference_steps"], s["guidance_scale"], s["batch_cfg"]) != (3, 1, False):
        raise ValueError("Require 3 DMD forwards / CFG1 / batch_cfg=false")
    if p["contract"] != dict(dmd_denoising_steps=[1000, 757, 522], flow_shift=8.0):
        raise ValueError("Require native FastWan DMD ladder and shift8")
    if p["vsa_sparsity"] != 0.9 or p["negative_prompt"] != "":
        raise ValueError("Require sparsity0.9 and empty negative prompt")
    allowed = {
        "performance_mode",
        "cpu_offload_components",
        "layerwise_offload_components",
        "pin_cpu_memory",
    }
    if set(p["server"]) - allowed:
        raise ValueError("Unsupported Wan server override")


def geometry(sampling: dict[str, Any]) -> tuple[list[int], int, int]:
    grid = [
        (sampling["num_frames"] - 1) // 4 + 1,
        sampling["height"] // 16,
        sampling["width"] // 16,
    ]
    blocks = math.prod(math.ceil(n / 4) for n in grid)
    return grid, blocks, max(1, math.ceil((1 - 0.9) * blocks))


def tensor_headers(path: Path) -> dict[str, Any]:
    tensors = {}
    for shard in sorted((path / "transformer").glob("*.safetensors")):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if (
                name in tensors
                or entry["data_offsets"][1] + size + 8 > shard.stat().st_size
            ):
                raise ValueError(f"Duplicate tensor or truncated shard: {name}")
            tensors[name] = dict(entry, file=str(shard))
    return tensors


def model_manifest(path, profile, gpu_counts):
    path = Path(path)
    verified = read_json(path / "download_verification.json")
    if (verified.get("model_id"), verified.get("revision"), verified.get("status")) != (
        MODEL_ID,
        REVISION,
        "complete",
    ):
        raise ValueError("Require a complete verified pinned FastWan snapshot")
    configs, weights = {}, {}
    for entry in verified["files"]:
        item = path / entry["file"]
        if not item.is_file() or item.stat().st_size != entry["size"]:
            raise ValueError(f"Missing/truncated pinned model asset: {item}")
        if item.suffix == ".json":
            configs[entry["file"]] = dict(sha256=sha256(item), config=read_json(item))
            if configs[entry["file"]]["sha256"] != entry["sha256"]:
                raise ValueError(f"Modified pinned config: {item}")
        elif item.suffix == ".safetensors":
            weights[entry["file"]] = item.stat().st_size
    if read_json(path / "model_index.json")["_class_name"] != "WanDMDPipeline":
        raise ValueError("Expected WanDMDPipeline")
    config = read_json(path / "transformer/config.json")
    if (
        config["num_layers"],
        config["num_attention_heads"],
        config["attention_head_dim"],
    ) != (40, 40, 128):
        raise ValueError("Unexpected 14B architecture")
    if any(n < 1 or 40 % n for n in gpu_counts):
        raise ValueError("GPU counts must divide 40 attention heads")
    tensors = tensor_headers(path)
    gates = {
        name: entry for name, entry in tensors.items() if ".to_gate_compress." in name
    }
    expected = {
        f"blocks.{i}.to_gate_compress.{kind}"
        for i in range(40)
        for kind in ["weight", "bias"]
    }
    if set(gates) != expected:
        raise ValueError("Incomplete learned VSA gates")
    for name, entry in gates.items():
        if entry["dtype"] != "BF16" or entry["shape"] != (
            [5120, 5120] if name.endswith("weight") else [5120]
        ):
            raise ValueError(f"Unexpected VSA gate geometry: {name}")
    return dict(
        model_id=MODEL_ID,
        revision=REVISION,
        configs=configs,
        weight_file_sizes=weights,
        download_verification=verified,
        transformer_tensors=len(tensors),
        trained_gate_count=40,
        trained_gate_tensors=len(gates),
        note="All downloaded LFS weights were SHA256 verified during preparation; launch checks sizes/configs.",
    )


def inspect_worker(worker, profile):
    import torch
    from safetensors import safe_open

    from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
        VSA_TILE_SIZE,
    )

    pipe = worker.pipeline
    model = pipe.get_module("transformer")
    config = worker.server_args.pipeline_config
    if type(pipe).__name__ != "WanDMDPipeline" or list(config.dmd_denoising_steps) != [
        1000,
        757,
        522,
    ]:
        raise ValueError("Expected native three-forward WanDMDPipeline")
    if config.flow_shift != 8.0 or pipe.get_module("scheduler").config.shift != 8.0:
        raise ValueError("Unexpected inference scheduler shift")
    stages = [s for s in pipe._stages if type(s).__name__ == "DmdDenoisingStage"]
    if len(stages) != 1 or stages[0].scheduler.config.shift != 8.0:
        raise ValueError("Unexpected DMD training-noise table shift")
    if (
        VSA_TILE_SIZE != (4, 4, 4)
        or worker.server_args.attention_backend_config["VSA_sparsity"] != 0.9
    ):
        raise ValueError("Unexpected VSA tile/sparsity")
    headers = tensor_headers(Path(worker.server_args.model_path))
    implementations = []
    count = 0
    for i, block in enumerate(model.blocks):
        if type(block).__name__ != "WanTransformerBlock_VSA":
            raise ValueError("DiT did not select native Wan VSA blocks")
        implementations.append(type(block.attn1.attn_impl).__name__)
        if implementations[-1] != "VideoSparseAttentionImpl":
            raise ValueError("Unexpected native attention implementation")
        for kind in ["weight", "bias"]:
            name = f"blocks.{i}.to_gate_compress.{kind}"
            value = block.to_gate_compress.get_parameter(kind)
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            if value.dtype != torch.bfloat16:
                raise ValueError("Expected BF16 gate weights")
            with safe_open(
                headers[name]["file"], framework="pt", device="cpu"
            ) as source:
                expected = source.get_tensor(name)
                if not torch.equal(value.detach().cpu(), expected):
                    raise ValueError(f"Loaded gate differs from checkpoint: {name}")
            count += 1
    if len(implementations) != 40 or count != 80:
        raise ValueError("Expected 40 VSA blocks and 80 gate tensors")
    grid, blocks, topk = geometry(profile["sampling"])
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return model, dict(
        rank=worker.rank,
        name=props.name,
        uuid=str(props.uuid),
        capability=[props.major, props.minor],
        total_memory_gib=props.total_memory / 2**30,
        pipeline_class=type(pipe).__name__,
        pipeline_config=asdict(config),
        server_args=asdict(worker.server_args),
        cpu_threads=torch.get_num_threads(),
        attention_implementations=implementations,
        verified_gate_tensors=count,
        trained_gate_count=40,
        video_grid=grid,
        video_blocks=blocks,
        topk=topk,
        vsa_tile_shape=list(VSA_TILE_SIZE),
        dit_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        vsa_dependency_build=read_json(ROOT / ".cache/benchmark/vsa_build.json")
        if (ROOT / ".cache/benchmark/vsa_build.json").exists()
        else None,
    )


class VSAExecutionAudit:
    """Observe native metadata; no tensor copies or GPU synchronization."""

    def __init__(self, model, profile):
        self.grid, self.blocks, self.topk = geometry(profile["sampling"])
        self.rows = []
        self.handle = model.register_forward_pre_hook(self.observe)

    def observe(self, module, inputs):
        from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
            _compute_cur_topk,
        )
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        context = get_forward_context()
        metadata = context.attn_metadata
        if (
            metadata.variable_block_sizes.numel() != self.blocks
            or list(metadata.dit_seq_shape) != self.grid
        ):
            raise ValueError("Unexpected actual VSA metadata shape")
        topk = _compute_cur_topk(metadata)
        if topk != self.topk:
            raise ValueError("Unexpected actual VSA topk")
        self.rows.append(
            dict(
                forward_index=context.current_timestep,
                video_blocks=self.blocks,
                topk=topk,
                sparsity=metadata.VSA_sparsity,
            )
        )

    def collect(self):
        if len(self.rows) != 3 or [r["forward_index"] for r in self.rows] != [0, 1, 2]:
            raise ValueError("Expected exactly three native VSA forwards")
        return list(self.rows)
