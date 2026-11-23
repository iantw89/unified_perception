# Training on a cloud GPU pod (RunPod H100)

The DINOv3 run (`configs/gym_sixcam_track_dinov3_cloud.yaml`) trains from a self-contained data
bundle copied to the pod's local disk.  Nothing streams from Google Drive during training: the
pipeline reads ~100k small files at random, which a Drive mount (API latency, request quotas)
cannot serve at training rate, so the bundle is downloaded once as a few large tar shards.

## 1. Build the caches (workstation)

| Cache | Tool | Notes |
|---|---|---|
| Gym, native 480 x 640 | `tools/import_golden.py --height 480 --width 640` | `data_cache/golden_20250505_480x640` |
| MOYO windows, native aspect | `tools/cache_eval_windows --config configs/gym_sixcam_track_dinov3.yaml --split val` then `--split train` | ~6 h (decodes from Drive) |
| Composition, compact | `tools/compact_composition_cache data/composition_640_c4_s8/cache` | 416 GB of NPY -> ~30 GB JPEG / bit-packed |

## 2. Pack the bundle onto Google Drive (workstation)

```
python -m unified_perception.tools.pack_cloud_bundle --config configs/gym_sixcam_track_dinov3.yaml --dry-run
python -m unified_perception.tools.pack_cloud_bundle --config configs/gym_sixcam_track_dinov3.yaml \
    --out "G:/My Drive/unified_perception_bundles/dinov3_native_v1"
```

Writes ~4 GB tar shards + `manifest.json` (sha256 per shard).  The packer refuses incomplete
caches.  Google Drive for desktop then uploads the folder (~80 GB; needs that much Drive space).

## 3. Pod

* **GPU**: 1 x H100 80 GB (PCIe is fine).  **vCPU**: 16+ (14 loader workers + the trainer).
* **Disk**: container disk >= 200 GB (local NVMe; repository + extracted data), plus a network
  volume at `/workspace` for checkpoints/TensorBoard (`runs/` is symlinked there) and, with
  `KEEP_BUNDLE=1`, the shards (so a new pod skips the Drive download).
* **Image**: any recent CUDA PyTorch image (e.g. `runpod/pytorch`); the script installs its own
  venv and picks the CUDA 13.0 or 12.8 torch build from the driver version.
* **Google Drive access** (rclone, configured by environment variables -- put the token in a
  RunPod *secret*, never in the repository):
  1. on your workstation: `rclone authorize drive` and copy the printed token JSON;
  2. pod env: `RCLONE_CONFIG_GDRIVE_TYPE=drive`, `RCLONE_CONFIG_GDRIVE_SCOPE=drive.readonly`,
     `RCLONE_CONFIG_GDRIVE_TOKEN={{ RUNPOD_SECRET_gdrive_token }}`,
     `BUNDLE_REMOTE=gdrive:unified_perception_bundles/dinov3_native_v1`.

Then on the pod:

```
curl -fsSL https://raw.githubusercontent.com/iantw89/unified_perception/main/cloud/setup_pod.sh -o setup_pod.sh
bash setup_pod.sh
```

It clones the repository, installs the environment, downloads + verifies + extracts the bundle
into the repository (members are repository-relative paths), links `runs/` to the volume, runs the
smoke tests and a short step benchmark, and prints the launch command:

```
python -m unified_perception.tools.training_process start --config configs/gym_sixcam_track_dinov3_cloud.yaml
```

Training is checkpoint-resumable (`last.pt` every 100 steps on the volume): on a new pod, rerun
`setup_pod.sh` and the same `start` command.

## Scaling notes

* Batch 16 (4 x the workstation run) with lr sqrt-scaled; the step is GPU-bound from ~12 lanes
  (benchmark 2026-11-23: host overhead ~0.6 s/step, independent of batch size).
* Worth measuring on the pod before a long run: `vit_compile` (on by default), TensorRT / FP8 for
  the frozen trunk (H100 FP8 GEMMs ~2 x bf16), and the auxiliary 2-D head (`model.aux2d_weight`,
  off by default) as an A/B.
