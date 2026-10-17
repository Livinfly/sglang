"""Rebuild the official Wan VSA extension for a local Torch/CUDA/GLIBC ABI.

Use --source-dir with the official FastVideo csrc/attn/video_sparse_attn
snapshot and its pinned ThunderKittens headers, and --cuda-home with a
consistent toolkit matching Torch. The installed vsa==0.0.4 Python package
is preserved; only its CUDA extension is rebuilt. No attention replacement.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    provenance = json.loads((source / "source_provenance.json").read_text())
    revision = "f8bfc760155c37c9ef89c0be90a3388b7a1a7da8"
    if provenance["revision"] != revision:
        raise ValueError("Unexpected official VSA source revision")
    for entry in provenance["tree"]:
        if entry["type"] != "blob":
            continue
        rel = entry["path"].split("csrc/attn/video_sparse_attn/", 1)[1]
        data = (source / rel).read_bytes()
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        if hashlib.sha1(blob).hexdigest() != entry["sha"]:
            raise ValueError(f"Modified official source: {rel}")
    from importlib.metadata import version

    import torch

    if version("vsa") != "0.0.4":
        raise ValueError("Install the native vsa==0.0.4 package first")
    installed = Path(sysconfig.get_path("platlib"))
    old_check = 'assert torch.__version__ >= "2.4.0"'
    new_check = (
        'assert tuple(int(n) for n in torch.__version__.split(".")[:2]) >= (2, 4)'
    )
    for name in ["__init__.py", "block_sparse_wrapper.py", "index.py"]:
        original = (source / "vsa" / name).read_text()
        actual = (installed / "vsa" / name).read_text()
        if actual not in {original, original.replace(old_check, new_check)}:
            raise ValueError(f"Native Python ABI differs: {name}")
    env = dict(os.environ, CUDA_HOME=str(args.cuda_home.resolve()), MAX_JOBS="4")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    # CUDA 13 expands -arch=sm_90a into extra generic sm_90 PTX, which
    # cannot represent WGMMA. Emit only the upstream kernel's sm_90a target.
    setup = (source / "setup.py").read_text()
    if setup.count("'-arch=sm_90a'") != 1:
        raise ValueError("Unexpected upstream architecture option")
    build_setup = source / "_local_build_setup.py"
    build_setup.write_text(
        setup.replace("'-arch=sm_90a'", "'-gencode=arch=compute_90a,code=sm_90a'")
    )
    try:
        subprocess.run(
            [sys.executable, str(build_setup), "build_ext", "--inplace"],
            cwd=source,
            env=env,
            check=True,
        )
    finally:
        build_setup.unlink()
    extension = next(source.glob("vsa_cuda*.so"))
    destination = installed / extension.name
    shutil.copy2(extension, destination)
    # Upstream compares version strings lexically, rejecting Torch 2.14 as
    # older than 2.4. Correct only the admission check, preserving all math.
    wrapper = installed / "vsa/block_sparse_wrapper.py"
    wrapper.write_text(wrapper.read_text().replace(old_check, new_check))
    result = dict(
        package="vsa==0.0.4",
        extension_source_revision=revision,
        thunderkittens_revision="6c27e28c8115d1839d9eeeb530913c184a75fc87",
        extension_sha256=sha256(destination),
        build_arch_option="-gencode=arch=compute_90a,code=sm_90a",
        python_compatibility_patch="Compare Torch major/minor numerically (2.14 >= 2.4). No kernel/math changes.",
        python_file_sha256={
            p.name: sha256(p) for p in (installed / "vsa").glob("*.py")
        },
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nvcc=subprocess.check_output(
            [str(args.cuda_home / "bin/nvcc"), "--version"], text=True
        ),
        reason="Official wheel requires GLIBC_2.32; rebuild upstream CUDA extension for local ABI. "
        "Preserve installed official Python implementation. Source snapshot is tagged 0.0.3; "
        "its Python forward/wrapper/index match 0.0.4. No attention algorithm replacement.",
    )
    write_json(ROOT / ".cache/benchmark/vsa_build.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
