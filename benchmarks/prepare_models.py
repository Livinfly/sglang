"""Convert a caller-supplied official FastH3 snapshot to SGLang's native layout.

The small upstream converter is pinned and hash checked. Tensor conversion runs
on CPU; source weights are never edited. DiT/TE weights are linked by default.
"""

import argparse
import importlib.util
import shutil
from pathlib import Path

from common import HERE, read_json, sha256, write_json
from config import model_manifest, validate_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overlay-path", type=Path, help="Offline copy of pinned overlay files"
    )
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy instead of symlink; output can be moved independently",
    )
    args = parser.parse_args()
    profile = read_json(args.config)
    validate_profile(profile)
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    pinned = read_json(HERE / "overlay.json")
    overlay = args.overlay_path
    if overlay is None:
        from huggingface_hub import snapshot_download

        overlay = Path(
            snapshot_download(
                pinned["repo"],
                revision=pinned["revision"],
                allow_patterns=list(pinned["files"]),
            )
        )
    for name, digest in pinned["files"].items():
        if sha256(overlay / name) != digest:
            raise ValueError(f"Overlay hash mismatch: {name}")
    mapping = read_json(overlay / "_overlay/overlay_manifest.json")
    for name in mapping["required_source_files"]:
        if not (source / name).is_file():
            raise ValueError(f"Missing original component: {name}")
    for component, shift in [
        ("scheduler", "video_shift"),
        ("audio_scheduler", "audio_shift"),
    ]:
        if (
            read_json(source / component / "scheduler_config.json")["shift"]
            != profile["contract"][shift]
        ):
            raise ValueError("Source checkpoint shift does not match selected profile")
    output.mkdir(parents=True)
    for item in mapping["file_mappings"]:
        for src in (source / item["src"]).rglob("*"):
            if not src.is_file() or ".cache" in src.parts:
                continue
            dst = output / item["dst_dir"] / src.relative_to(source / item["src"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".safetensors" and not args.copy_weights:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
    spec = importlib.util.spec_from_file_location(
        "pinned_fasth3_overlay", overlay / "_overlay/materialize.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    converter.materialize(
        overlay_dir=str(overlay),
        source_dir=str(source),
        output_dir=str(output),
        manifest=mapping,
    )
    index = read_json(overlay / "model_index.json")
    index["_class_name"] = (
        "FastH3V1BaselinePipeline" if profile["variant"] == "v1" else "FastH3Pipeline"
    )
    index["_minimax_h3"]["sigma_shift_scales"] = dict(
        video=profile["contract"]["video_shift"],
        audio=profile["contract"]["audio_shift"],
    )
    index["_minimax_h3"]["dmd_denoising_steps"] = profile["contract"][
        "dmd_denoising_steps"
    ]
    write_json(output / "model_index.json", index)
    model_manifest(output, profile, [1])
    write_json(
        output / "materialization_complete.json",
        dict(
            source=str(source),
            overlay=pinned,
            profile=profile["name"],
            linked_weights=not args.copy_weights,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
