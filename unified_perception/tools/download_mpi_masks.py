"""Fetch only official MPI masks; never replace RGB, annotations, or existing masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

from ..data.composition import file_hash
from .compose import write_json

BASE_URL = "https://vcai.mpi-inf.mpg.de/3dhp-dataset"


def install_archive(archive: Path, output: Path) -> dict[str, str]:
    """Extract only camera AVIs, flattening the official archive's directory prefixes."""
    output.mkdir(parents=True, exist_ok=True)
    installed = {}
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/").split("/")[-1]
            if not re.fullmatch(r"video_(?:[0-9]|1[0-3])\.avi", name):
                raise ValueError(f"Unexpected MPI mask archive member: {member.filename}")
            if name in installed:
                raise ValueError(f"Duplicate camera video in MPI archive: {name}")
            destination = output / name
            with tempfile.TemporaryDirectory(dir=output) as tmp:
                candidate = Path(tmp) / name
                with zipped.open(member) as source, candidate.open("wb") as target:
                    shutil.copyfileobj(source, target)
                digest = file_hash(candidate)
                if destination.exists():
                    if file_hash(destination) != digest:
                        raise FileExistsError(f"Existing MPI mask differs: {destination}")
                else:
                    candidate.replace(destination)
            installed[name] = digest
    if not installed:
        raise ValueError("No camera videos in MPI mask archive")
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1, 9), default=list(range(1, 9)))
    parser.add_argument("--sequences", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--archives", nargs="+", choices=("vnect_cameras", "other_angled_cameras", "ceiling_cameras"),
                        default=["vnect_cameras"])
    parser.add_argument("--accept-dataset-license", action="store_true",
                        help="Confirm you have read and accepted the MPI-INF-3DHP dataset license")
    args = parser.parse_args()
    if not args.accept_dataset_license:
        parser.error("Read the MPI-INF-3DHP dataset license, then pass --accept-dataset-license")
    for subject in args.subjects:
        for sequence in args.sequences:
            for kind in ("FGmasks", "ChairMasks"):
                folder = args.root / f"S{subject}" / f"Seq{sequence}" / kind
                for archive in args.archives:
                    receipt = folder / f"{archive}-download.json"
                    if receipt.exists():
                        record = json.loads(receipt.read_text())
                        if not record["files"] or any(
                            not (folder / name).is_file() or file_hash(folder / name) != digest
                            for name, digest in record["files"].items()
                        ):
                            raise ValueError(f"Downloaded MPI masks changed: {receipt}")
                        print(f"[verified] {folder} {archive}", flush=True)
                        continue
                    url = f"{BASE_URL}/S{subject}/Seq{sequence}/{kind}/{archive}.zip"
                    print(f"[download] {url}", flush=True)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "masks.zip"
                        with urlopen(url, timeout=120) as response, path.open("wb") as target:
                            shutil.copyfileobj(response, target)
                        files = install_archive(path, folder)
                        write_json(receipt, dict(url=url, archive_sha256=file_hash(path), files=files))


if __name__ == "__main__":
    main()
