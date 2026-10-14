"""Lightweight config system.

A single nested dataclass tree describes every hyper-parameter in the project.
Configs are created in code (sensible defaults live here) and can be overridden
by a YAML file and/or ``key=value`` command-line style overrides.  There is no
registry, no builder, and no import-by-string machinery -- everything is plain
Python.

Usage
-----
    from unified_perception.config import Config, load_config

    cfg = load_config("configs/synthetic_small.yaml",
                      overrides=["train.max_epochs=5", "data.batch_size=4"])
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # Data source provider
    provider: str = "human_prior_aligned"
    image_height: int = 256
    image_width: int = 256
    # Number of camera views per frame.
    num_cameras: int = 4
    # Temporal sequence length (frames per training sample).
    seq_len: int = 4
    batch_size: int = 4
    num_workers: int = 0
    # Max people in a single frame (used to size GT tensors; data may have fewer).
    max_people: int = 8
    # Sliding-window stride across frames.
    stride: int = 2
    # If > 0, cap the number of training samples per epoch (useful for pipe-cleaner runs).
    max_samples_per_epoch: int = 0  # 0 = no limit
    # If true, decode video frames on the GPU (requires torchcodec with GPU support).
    gpu_decode: bool = False

    # Fraction of discovered sequences assigned to validation for providers that
    # discover raw sequence folders directly (human_prior_aligned).
    val_fraction: float = 0.2
    split_seed: int = 0
    # YOLO-compatible normalisation: RGB uint8 [0, 255] → float [0, 1]  (÷255).
    # YOLOPoseBackbone expects [0, 1] input; ImageNet-style mean/std shifts
    # activations outside the trained regime and breaks predict_2d.
    pixel_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pixel_std: List[float] = field(default_factory=lambda: [255.0, 255.0, 255.0])
    # Training-time augmentation.
    aug_flip_views: bool = False  # rendered views are calibrated; flipping is unsafe

    # --- human_prior aligned provider settings ---
    # Path to sibling human_prior repository (used to import dataset config readers).
    human_prior_repo: str = "../human_prior"
    # Sequence output folders produced by human_prior alignment pipeline.
    # Each sequence dir should contain mocap_keypoints.json and alignment_results.json.
    human_prior_aligned_dirs: List[str] = field(default_factory=list)
    # Target FPS used when generating alignment outputs; provider mirrors this
    # camera-frame skipping policy when selecting video frames.
    human_prior_target_fps: float = 10.0
    # Raw dataset roots (used by human_prior dataset adapters to fetch images/calib).
    human_prior_h36m_dataset_path: Optional[str] = None
    human_prior_moyo_dataset_path: Optional[str] = None
    human_prior_mpi_dataset_path: Optional[str] = None

    # --- Shuffle buffer (IterableDataset mode) ---
    # When True, use ShuffleBufferDataset instead of MultiViewPoseDataset.
    # The buffer fully decodes randomly-selected sequences into memory and serves
    # windows by array-slicing, eliminating per-window video-seek overhead.
    use_shuffle_buffer: bool = True
    # Total camera-frames to keep in the in-memory pool per worker.
    # One camera-frame ≈ 3 × H × W bytes (uint8).  At 640×640: ~1.2 MB / frame.
    # 2048 frames ≈ 2.5 GB, 4096 ≈ 5 GB.  Larger values improve shuffle diversity.
    buffer_frames: int = 2048
    # Minimum number of unique segments to keep loaded in the pool simultaneously.
    # Enforces shuffle diversity by preventing one large segment from dominating.
    # Violated only when fewer segments are available.
    min_buffer_seg: int = 4
    # Number of decoded chunks the background decode thread can stage in the
    # prefetch queue before it blocks (back-pressure).  Increasing this reduces
    # stall latency spikes at the cost of extra memory:
    #   extra_bytes ≈ prefetch_queue_depth × (buffer_frames/min_buffer_seg) × H × W × 3
    # 0 means "auto" → uses min_buffer_seg.
    prefetch_queue_depth: int = 0


@dataclass
class ModelConfig:
    embed_dims: int = 256
    # Position range (capture volume) in millimetres: [xmin,ymin,zmin,xmax,ymax,zmax].
    position_range: List[float] = field(
        default_factory=lambda: [-6000.0, -6000.0, -200.0, 6000.0, 6000.0, 3000.0]
    )
    num_joints: int = 12

    # --- Image backbone ---
    backbone_type: str = "yolo_pose"  # "yolo_pose" | "vit"

    # YOLO-pose backbone (from ultralytics PoseModel, pre-head feature map tap).
    yolo_model_name: str = "yolo11n-pose.pt"
    # Layers to tap for multi-scale features (default: layers 16, 19, 22).
    yolo_feature_indices: List[int] = field(default_factory=lambda: [16, 19, 22])
    yolo_trainable: bool = True
    yolo_force_download: bool = True

    # --- 3D position encoder ---
    depth_num: int = 64
    depth_start_mm: float = 500.0
    depth_max_mm: float = 8000.0
    lid_depth: bool = True  # log-increasing depth bins (LID) vs uniform

    # --- Transformer decoder ---
    num_queries: int = 128
    dec_layers: int = 6
    dec_heads: int = 8
    dec_ffn_dim: int = 512
    # Sampling points per camera per query in Detr3DCrossAttn.
    # StreamPETR uses 13 (1 reference + 12 offsets)
    dec_num_pts: int = 13
    # Feature-aggregation groups in Detr3DCrossAttn, matching StreamPETR's
    # DeformableFeatureAggregationCuda ``num_groups``.  Must divide embed_dims
    # evenly.  1 = single-group (default, matches previous behaviour).
    dec_groups: int = 1
    dropout: float = 0.1

    # --- Temporal streaming memory ---
    num_propagated: int = 16  # queries carried over from the previous frame
    memory_topk: int = 16  # top-k confident queries pushed to the memory bank
    memory_len: int = 256  # max length of the streaming memory bank

    # --- Query denoising (DN, training only) ---
    with_dn: bool = True
    dn_scalar: int = 10  # number of noised GT groups
    dn_noise_scale: float = 0.2  # noise magnitude on the normalised anchor
    dn_split: float = 0.75  # positive rate: keep query as person when noise norm <= split
    dn_weight: float = 1.0  # weight on the DN loss terms

    # --- Experimental / optional backends ---
    # When True, uses the MMCV CUDA MSDA kernel instead of F.grid_sample.
    # Requires mmcv with CUDA extensions installed.  No-op (falls back to
    # PyTorch) when mmcv is not available or device is CPU.
    use_mmcv_msda: bool = True

    # --- Loss / matcher weights ---
    cost_class: float = 2.0
    cost_kpt: float = 5.0
    loss_class: float = 2.0
    loss_kpt: float = 5.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    max_epochs: int = 20
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 35.0
    warmup_iters: int = 100
    log_interval: int = 10
    ckpt_interval: int = 1  # epochs
    eval_interval: int = 1  # epochs
    viz_batch_interval: int = 0  # save training viz every N global steps (0 = off)
    device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps"
    seed: int = 0
    resume: Optional[str] = None
    # torch.compile the model before training for improved throughput.
    # Disabled by default; enable via config or --compile CLI flag on the profiler.
    # The first few steps will be slower while kernels are compiled and cached.
    compile_model: bool = True


@dataclass
class EvalConfig:
    # MPJPE thresholds (mm) for PCK-style accuracy reporting.
    pck_thresholds_mm: List[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    # If > 0, cap the number of validation batches (useful for pipe-cleaner runs).
    max_eval_batches: int = 0  # 0 = no limit

    # A predicted person matches a GT person if its root distance is below this.
    match_threshold_mm: float = 500.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# --------------------------------------------------------------------------- #
# Loading / merging                                                           #
# --------------------------------------------------------------------------- #
def _merge_dict(cfg_obj: Any, updates: dict) -> None:
    """Recursively assign values from ``updates`` onto a dataclass instance."""
    for key, value in updates.items():
        if not hasattr(cfg_obj, key):
            raise KeyError(f"Unknown config key: {key!r} on {type(cfg_obj).__name__}")
        current = getattr(cfg_obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dict(current, value)
        else:
            setattr(cfg_obj, key, value)


def _apply_override(cfg: Config, override: str) -> None:
    """Apply a single ``a.b.c=value`` override string."""
    if "=" not in override:
        raise ValueError(f"Override must be 'key.path=value', got {override!r}")
    path, raw_value = override.split("=", 1)
    keys = path.split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    leaf = keys[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"Unknown config key: {path!r}")
    try:
        value = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        value = raw_value  # plain string
    setattr(obj, leaf, value)


def load_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Config:
    """Build a :class:`Config`, optionally merging a YAML file and CLI overrides."""
    cfg = Config()
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        _merge_dict(cfg, data)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
