"""Small, CPU-only workload and native-export checks."""

import math
import re
from pathlib import Path

from common import read_json, sha256


def validate_profile(p):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", p["name"]):
        raise ValueError("Unsafe profile name")
    if p["adapter"] == "fastwan21_14b":
        from fastwan import validate_profile as validate_wan

        return validate_wan(p)
    if p["adapter"] != "minimax_h3" or p["variant"] not in {"v1", "v2"}:
        raise ValueError("Supported profiles: MiniMax H3 FastH3 V1/V2")
    if p["negative_prompt"] not in {"", None}:
        raise ValueError("Native FastH3 has no negative prompt")
    s = p["sampling"]
    steps = 4 if p["variant"] == "v1" else 8
    if (
        s["num_inference_steps"] != steps
        or len(p["contract"]["dmd_denoising_steps"]) != steps
    ):
        raise ValueError(
            "SGLang counts DiT forwards, not sigma grid points: V1=4, V2=8"
        )
    if s["guidance_scale"] != 1 or s["task"] != "t2va" or s["fps"] != 24:
        raise ValueError("FastH3 requires t2va, CFG=1 and 24 fps")
    for key in ["width", "height", "num_frames"]:
        if type(s[key]) is not int or s[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if s["quality"] not in {"lossless", "exact"}:
        raise ValueError("FastH3 supports lossless/exact quality")
    if not math.isfinite(p["vsa_sparsity"]) or not 0 < p["vsa_sparsity"] < 1:
        raise ValueError("Invalid VSA sparsity")
    # Topology/timing boundaries must remain controlled by this harness.
    allowed = {
        "performance_mode",
        "cpu_offload_components",
        "dit_cpu_offload",
        "text_encoder_cpu_offload",
        "vae_cpu_offload",
        "dit_layerwise_offload",
        "layerwise_offload_components",
        "pin_cpu_memory",
        "use_fsdp_inference",
        "encoder_parallel",
        "disable_conditioning_cache",
    }
    if set(p["server"]) - allowed:
        raise ValueError(
            "Unsupported server override: " + str(set(p["server"]) - allowed)
        )


def model_manifest(path, profile, gpu_counts):
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import model_manifest as wan_manifest

        return wan_manifest(path, profile, gpu_counts)
    path = Path(path)
    index = read_json(path / "model_index.json")
    native = index["_minimax_h3"]
    contract = profile["contract"]
    if native["dmd_denoising_steps"] != contract["dmd_denoising_steps"] or native[
        "sigma_shift_scales"
    ] != {"video": contract["video_shift"], "audio": contract["audio_shift"]}:
        raise ValueError("Native model schedule/shifts do not match profile")
    config = read_json(path / "transformer/config.json")
    # This pinned native H3 implementation packs sequences to 64-token boundaries.
    if any(config["num_attention_heads"] % n or 64 % n for n in gpu_counts):
        raise ValueError(
            "GPU counts must divide attention heads and native 64-token alignment"
        )
    weights = read_json(
        path / "transformer/diffusion_pytorch_model.safetensors.index.json"
    )["weight_map"]
    gates = [k for k in weights if "to_gate_compress" in k and k.endswith(".weight")]
    if len(gates) != config["num_layers"]:
        raise ValueError("Checkpoint must contain every trained VSA compression gate")
    for component in [
        "transformer",
        "text_encoder",
        "audio_vae",
        "video_vae",
        "tokenizer",
        "processor",
    ]:
        folder = path / component
        if not folder.is_dir() or not any(folder.iterdir()):
            raise ValueError(f"Missing component: {folder}")
        for idx in folder.glob("*.index.json"):
            for shard in set(read_json(idx).get("weight_map", {}).values()):
                if not (folder / shard).is_file():
                    raise ValueError(f"Missing shard: {folder / shard}")
    for name in [
        "video_vae/source/model.safetensors",
        "transformer/sglang_rope_inv_freq.safetensors",
    ]:
        if not (path / name).is_file():
            raise ValueError(f"Missing native export: {name}; use prepare_models.py")
    configs, sizes = {}, {}
    for item in path.rglob("*"):
        if not item.is_file() or ".cache" in item.parts:
            continue
        name = str(item.relative_to(path))
        if item.suffix == ".json" and not item.name.endswith(".index.json"):
            configs[name] = dict(sha256=sha256(item), config=read_json(item))
        elif item.suffix in {".safetensors", ".bin", ".pt"}:
            sizes[name] = item.stat().st_size
    return dict(
        configs=configs,
        weight_file_sizes=sizes,
        trained_gate_count=len(gates),
        note="Config hashes, weight sizes and index presence; weight contents not fully hashed.",
    )
