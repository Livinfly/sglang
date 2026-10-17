"""Opt-in, hash-verified ffmpeg and CUDA forward-compatibility downloads."""

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from common import HERE, ROOT, read_json, sha256


def download(url, path, expected):
    if path.exists() and sha256(path) == expected:
        return
    temporary = path.with_suffix(".download")
    urllib.request.urlretrieve(url, temporary)
    if sha256(temporary) != expected:
        raise ValueError(f"Download checksum mismatch: {url}")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--cuda-compat", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/benchmark/tools")
    args = parser.parse_args()
    if not args.media and not args.cuda_compat:
        parser.error("Select --media and/or --cuda-compat")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cuda_compat:
        manifest = read_json(HERE / "cuda_compat.json")
        archive = args.output / "cuda-compat.deb"
        download(manifest["url"], archive, manifest["sha256"])
        subprocess.run(
            ["dpkg-deb", "-x", str(archive), str(args.output / "cuda-compat")],
            check=True,
        )
        print(
            "Set BENCH_CUDA_COMPAT to:",
            args.output / "cuda-compat/usr/local/cuda-13.0/compat",
        )
    if args.media:
        manifest = read_json(HERE / "media_tools.json")
        archive = args.output / "ffmpeg.tar.xz"
        download(manifest["url"], archive, manifest["archive_sha256"])
        (args.output / "bin").mkdir(exist_ok=True)
        with tarfile.open(archive) as bundle:
            for name, expected in manifest["binaries"].items():
                members = [
                    m
                    for m in bundle.getmembers()
                    if m.isfile() and Path(m.name).name == name
                ]
                if len(members) != 1:
                    raise ValueError("Ambiguous media executable")
                target = args.output / "bin" / name
                with bundle.extractfile(members[0]) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(target) != expected:
                    raise ValueError(f"Media checksum mismatch: {name}")
                target.chmod(0o755)


if __name__ == "__main__":
    main()
