"""Pack everything a training run reads into tar shards for a cloud pod (cloud/README.md).

    python -m unified_perception.tools.pack_cloud_bundle --config configs/gym_sixcam_track_dinov3.yaml \\
        --out "G:/My Drive/unified_perception_bundles/dinov3_native_v1"

Members are stored at their repository-relative paths, so ``tar -xf`` in the repository root of
a fresh clone reproduces the layout the cloud config expects:

* the composition cache, compact pixels only (tools/compact_composition_cache.py; NPY pixel files
  and render backgrounds are not needed for training);
* the gym cache (``data.gym_pseudo3d_root``);
* the aligned (MOYO) window caches for train and val (``data.eval_window_cache``), which must be
  complete -- the pod has no source videos;
* the aligned sequences' metadata (JSON only) as ``data_cache/moyo_meta/<folder>/...``;
* the registered backbone weights and the external re-ID model.

Shards are uncompressed tars of ~``--shard-gb`` (the payload is JPEG/NPZ already), written next to
``manifest.json`` (sha256 and size per shard), so a pod can verify before extracting.  Refuses to
pack incomplete caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path, PureWindowsPath

from ..config import load_config
from ..data.composition import compact_path
from ..models.backbones import BACKBONES, REPO_ROOT

MOYO_META = "data_cache/moyo_meta"


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def collect(cfg) -> list[tuple[Path, str]]:
    """(source path, archive name) for every file the run needs."""
    from ..data.streaming import aligned_component
    members: list[tuple[Path, str]] = []
    manifest = (REPO_ROOT / cfg.data.composition_manifest).resolve()
    comp_root = manifest.parent.parent                       # data/<experiment>/ (cache/ inside)
    missing = []
    for path in sorted(p for p in comp_root.rglob("*") if p.is_file()):
        name = path.name
        if name.endswith(".npy"):
            if not compact_path(path).exists():
                missing.append(path)
            continue                                         # shipped as its compact sibling
        if name.endswith("-background.npz") or name.endswith(".tmp"):
            continue                                         # rendering inputs / partial writes
        members.append((path, _rel(path)))
    if missing:
        raise SystemExit(f"{len(missing)} composition NPY files have no compact sibling yet, e.g. {missing[0]} "
                         "(run tools/compact_composition_cache.py first)")
    gym = (REPO_ROOT / cfg.data.gym_pseudo3d_root).resolve()
    members += [(p, _rel(p)) for p in sorted(gym.rglob("*")) if p.is_file()]
    if cfg.data.gym_mix_aligned_dirs and cfg.data.gym_mix_aligned_fraction > 0:
        for split in ("val", "train"):
            component = aligned_component(cfg, split, cache=True)
            keys = [k for g in component.groups for k in g]
            absent = [k for k in keys if not component.load.path(k).exists()]
            if absent:
                raise SystemExit(f"aligned {split} cache incomplete: {len(absent)}/{len(keys)} windows missing "
                                 f"under {component.load.root} (run cache_eval_windows --split {split})")
            root = component.load.root
            members += [(p, _rel(p)) for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")]
        for source in cfg.data.gym_mix_aligned_dirs:
            folder = Path(source)
            for path in sorted(folder.glob("*/*.json")):
                members.append((path, f"{MOYO_META}/{PureWindowsPath(source).name}/{path.parent.name}/{path.name}"))
    if cfg.model.backbone:
        weights = (REPO_ROOT / BACKBONES[cfg.model.backbone].path).resolve()
        members.append((weights, _rel(weights)))
    if cfg.model.reid_external_model:
        reid = (REPO_ROOT / cfg.model.reid_external_model).resolve()
        members += [(p, _rel(p)) for p in (reid, reid.with_suffix(".json")) if p.is_file()]
    return members


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the local training config (source paths)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true", help="list sizes only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    members = collect(cfg)
    total = sum(p.stat().st_size for p, _ in members)
    groups: dict[str, int] = {}
    for p, name in members:
        key = "/".join(name.split("/")[:2])
        groups[key] = groups.get(key, 0) + p.stat().st_size
    for key, size in sorted(groups.items()):
        print(f"  {size / 1e9:8.2f} GB  {key}")
    print(f"{len(members)} files, {total / 1e9:.1f} GB", flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    limit, shards, index, start = args.shard_gb * 1e9, [], 0, time.perf_counter()
    while index < len(members):
        name = f"bundle-{len(shards):03d}.tar"
        path, size, count = args.out / name, 0, 0
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
            while index < len(members) and (count == 0 or size < limit):
                source, arcname = members[index]
                tar.add(str(source), arcname=arcname, recursive=False)
                size, count, index = size + source.stat().st_size, count + 1, index + 1
        shards.append({"name": name, "bytes": path.stat().st_size, "files": count, "sha256": _sha256(path)})
        print(f"  {name}: {count} files, {shards[-1]['bytes'] / 1e9:.2f} GB "
              f"({index}/{len(members)}, {time.perf_counter() - start:.0f} s)", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    (args.out / "manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit, "config": args.config,
        "files": len(members), "bytes": total, "groups_bytes": groups, "shards": shards}, indent=2))
    print(f"wrote {len(shards)} shards + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
