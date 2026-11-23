"""Registered image backbones.

Select one with ``model.backbone: <name>``; it sets the checkpoint path and the
tapped feature layers and verifies the checkpoint's SHA-256 so a run can never
silently pick up retrained weights at the same path.  Relative paths resolve
against the repository root (the parent of the ``unified_perception`` package).
``model.yolo_model_name`` remains available for unregistered experiments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class YoloBackbone:
    path: str
    sha256: str
    feature_indices: tuple[int, ...]
    description: str
    #: "yolo_pose" (tapped ultralytics layers) or "vit" (DINOv3 + SimpleFPN, models/vit_backbone.py;
    #: ``arch`` names the ViT size).
    kind: str = "yolo_pose"
    arch: str = ""


# YOLOv8x-pose-p6 head inputs (strides 8/16/32/64).
_P6_FEATURES = (20, 23, 26, 29)

BACKBONES: dict[str, YoloBackbone] = {
    "yolov8x_pose_p6": YoloBackbone(
        "../cvat_labeling/finetuning/weights/yolov8x-pose-p6.pt",
        "319004a4a2fe2da6734f881fe628f5825e5864f59a990e0faa90f7cbe30f4842",
        _P6_FEATURES, "Ultralytics pretrained 1280px baseline (no domain fine-tuning)"),
    "yolov8x_pose_p6_curated_v6": YoloBackbone(
        "../cvat_labeling/finetuning/runs/curated_v6/weights/best.pt",
        "83a7b40254ee5c4d740479e385a75d9f0c724b94f1471e3c877db845eee40f4c",
        _P6_FEATURES, "gym reviewed tasks (6 cameras); used by gym_sixcam_yolov8x_c7"),
    "yolov8x_pose_p6_curated_v7_al": YoloBackbone(
        "../cvat_labeling/finetuning/runs/curated_v7_al/weights/best.pt",
        "8dd7315144cf47a129f0968c2bf56e0d242a20c989bb507e14db0e78425e1343",
        _P6_FEATURES, "curated_v6 data + accepted active-learning round-1 jobs (task 247); "
                      "fine-tuned from the pretrained 1280px baseline"),
    # DINOv3 ViT-B/16 distilled on LVD-1689M (DINOv3 License; gated download, weights/ is not
    # committed).  Strides 8/16/32 (optionally 64) through the SimpleFPN.
    "dinov3_vitb16_lvd1689m": YoloBackbone(
        "weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
        (), "DINOv3 ViT-B/16 LVD-1689M pretrain (frozen) + ViTDet SimpleFPN", kind="vit", arch="vitb16"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_backbone(name: str, verify: bool = True) -> tuple[str, list[int]]:
    """Return (absolute checkpoint path, feature indices) for a registered backbone."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name!r}. Registered: {', '.join(sorted(BACKBONES))}")
    entry = BACKBONES[name]
    path = Path(entry.path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Backbone {name!r} checkpoint missing: {path}")
    if verify and file_sha256(path) != entry.sha256:
        raise RuntimeError(f"Backbone {name!r} checkpoint hash differs from the registry: {path}")
    return str(path), list(entry.feature_indices)


def resolve_repo_path(path: str) -> str:
    """Absolute path; relative paths resolve against the repository root."""
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve())
