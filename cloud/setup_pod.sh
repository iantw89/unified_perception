#!/usr/bin/env bash
# Bring up a Linux GPU pod (e.g. RunPod H100) for unified_perception training.  See cloud/README.md.
#
#   bash setup_pod.sh            # clone/update, install, fetch + verify + extract the data bundle, smoke-test
#
# Environment (RunPod: template env vars / secrets):
#   BUNDLE_REMOTE   rclone path of the bundle folder, e.g. gdrive:unified_perception_bundles/dinov3_native_v1
#                   (the rclone remote itself is configured by RCLONE_CONFIG_<NAME>_* variables, e.g.
#                   RCLONE_CONFIG_GDRIVE_TYPE=drive, RCLONE_CONFIG_GDRIVE_TOKEN=<from `rclone authorize drive`>)
#   REPO_URL        default https://github.com/iantw89/unified_perception
#   REPO_BRANCH     default main
#   WORK            local NVMe working directory (repository + data), default /root/up
#   PERSIST         persistent volume for runs/ and the downloaded bundle, default /workspace
#   KEEP_BUNDLE     1 = keep the tar shards under $PERSIST/bundles (skip the download next time), default 1
#   CONFIG          training config, default configs/gym_sixcam_track_dinov3_cloud.yaml
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/iantw89/unified_perception}
REPO_BRANCH=${REPO_BRANCH:-main}
WORK=${WORK:-/root/up}
PERSIST=${PERSIST:-/workspace}
KEEP_BUNDLE=${KEEP_BUNDLE:-1}
CONFIG=${CONFIG:-configs/gym_sixcam_track_dinov3_cloud.yaml}
REPO="$WORK/unified_perception"
log() { printf '\n== %s\n' "$*"; }

log "GPU / driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
# The CUDA 13.0 torch / mmcv wheels need driver >= 580 (create the pod with minCudaVersion 13.0).
if [ "$DRIVER_MAJOR" -lt 580 ]; then echo "driver $DRIVER_MAJOR < 580: CUDA 13.0 wheels will not run" >&2; exit 1; fi
mkdir -p "$WORK"
echo "$(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h "$WORK" | tail -1 | awk '{print $4}') free on WORK"

log "Repository"
mkdir -p "$WORK"
if [ -d "$REPO/.git" ]; then git -C "$REPO" fetch -q origin "$REPO_BRANCH" && git -C "$REPO" checkout -q "$REPO_BRANCH" \
    && git -C "$REPO" reset -q --hard "origin/$REPO_BRANCH"
else git clone -q --branch "$REPO_BRANCH" "$REPO_URL" "$REPO"; fi
git -C "$REPO" log --oneline -1

log "Python environment"
cd "$REPO"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
# Same versions as the workstation (torch 2.12.1); the mmcv wheel below is built against 2.12 / cu130.
TORCH_INDEX=https://download.pytorch.org/whl/cu130
pip install -q torch==2.12.1 torchvision==0.27.1 torchcodec==0.14.0 --index-url "$TORCH_INDEX"
# requirements.txt pins the Windows workstation's torch index; install the rest from PyPI.
grep -v -E '^--index-url|^(torch|torchvision|torchcodec)([<>=;[:space:]]|$)' requirements.txt > /tmp/requirements-pod.txt
pip install -q -r /tmp/requirements-pod.txt
# models/transformer.py imports mmcv (multi-scale deformable attention CUDA op): prebuilt wheel,
# the same build the workstation uses (MiroPsota/torch_packages_builder, mmcv main @ a8073c7).
pip install -q mmengine==0.10.7 \
    "https://github.com/MiroPsota/torch_packages_builder/releases/download/mmcv-2.2.0%2Ba8073c7/mmcv-2.2.0%2Ba8073c7pt2.12.0cu130-cp312-cp312-linux_x86_64.whl"
python -c "from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction; print('mmcv MSDA ok')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

log "Data bundle"
if [ -z "${BUNDLE_REMOTE:-}" ]; then echo "BUNDLE_REMOTE is not set" >&2; exit 1; fi
command -v rclone >/dev/null || (apt-get update -qq && apt-get install -y -qq rclone)
SHARDS="$PERSIST/bundles/$(basename "$BUNDLE_REMOTE")"
[ "$KEEP_BUNDLE" = 1 ] || SHARDS="$WORK/bundle"
mkdir -p "$SHARDS"
# Many parallel transfers: Drive throughput is per file stream.
rclone copy "$BUNDLE_REMOTE" "$SHARDS" --transfers 8 --checkers 16 --drive-chunk-size 256M --progress --stats-one-line
python - "$SHARDS" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); manifest = json.loads((root / "manifest.json").read_text())
for shard in manifest["shards"]:
    path = root / shard["name"]; h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    if path.stat().st_size != shard["bytes"] or h.hexdigest() != shard["sha256"]:
        sys.exit(f"corrupt shard {path}")
print(f"verified {len(manifest['shards'])} shards, {manifest['bytes'] / 1e9:.1f} GB, packed at commit {manifest['git_commit'][:10]}")
PY
# Without KEEP_BUNDLE each shard is deleted once extracted (peak disk ~ bundle size + one shard).
for shard in "$SHARDS"/bundle-*.tar; do
    tar -xf "$shard" -C "$REPO"
    [ "$KEEP_BUNDLE" = 1 ] || rm -f "$shard"
done
[ "$KEEP_BUNDLE" = 1 ] || rm -rf "$SHARDS"

log "Persistent run outputs"
mkdir -p "$PERSIST/runs"
if [ ! -L "$REPO/runs" ]; then rm -rf "$REPO/runs"; ln -s "$PERSIST/runs" "$REPO/runs"; fi

log "Smoke tests"
python -m pytest -q -p no:warnings tests/test_vit_backbone.py tests/test_gpu_assignment.py tests/test_pose2d_aux.py \
    tests/test_composition_compact.py tests/test_deferred_step.py
python -m unified_perception.tools.bench_step "$CONFIG" --batches 8 --warmup 3 --steps 8 --deferred

cat <<EOF

Ready.  Launch (detached, resumable; logs in runs/<name>/console.log):
  cd $REPO && . .venv/bin/activate
  python -m unified_perception.tools.training_process start --config $CONFIG
  python -m unified_perception.tools.training_process status --config $CONFIG
TensorBoard:  tensorboard --logdir runs --host 0.0.0.0 --port 6006   (expose port 6006 on the pod)
EOF
