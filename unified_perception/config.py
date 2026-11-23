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
    # Treat num_cameras as a ceiling; collate masks padding for mixed-view batches.
    allow_fewer_cameras: bool = False
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
    # YOLO-compatible normalisation: RGB uint8 [0, 255] â†’ float [0, 1]  (Ã·255).
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

    # --- Reviewed sparse gym cache and original-mocap mixture ---
    gym_pseudo3d_root: str = "data_cache/gym_pseudo3d_20250505"
    gym_sample_fraction: float = 0.25
    # Optional single-person aligned recordings (e.g. MOYO, which composition
    # excludes) read through the human_prior_aligned provider as a third mixture
    # component holding this fraction of all mix samples (0 = off).
    gym_mix_aligned_dirs: List[str] = field(default_factory=list)
    gym_mix_aligned_fraction: float = 0.0
    # Local disk cache of decoded evaluation windows for slow (video-decoded) sources, e.g.
    # val_aligned (data/streaming.py: CachedWindows).  Empty = off.  A subfolder per content
    # fingerprint; prebuild with  python -m unified_perception.tools.cache_eval_windows.
    eval_window_cache: str = ""
    # Serve the *training* aligned windows through the same cache (folder per split).  Misses
    # decode from the source and fill the cache; prebuild with cache_eval_windows --split train.
    train_window_cache: bool = False

    # --- Shuffle buffer (IterableDataset mode) ---
    # When True, use ShuffleBufferDataset instead of MultiViewPoseDataset.
    # The buffer fully decodes randomly-selected sequences into memory and serves
    # windows by array-slicing, eliminating per-window video-seek overhead.
    use_shuffle_buffer: bool = True
    # Total camera-frames to keep in the in-memory pool per worker.
    # One camera-frame â‰ˆ 3 Ã— H Ã— W bytes (uint8).  At 640Ã—640: ~1.2 MB / frame.
    # 2048 frames â‰ˆ 2.5 GB, 4096 â‰ˆ 5 GB.  Larger values improve shuffle diversity.
    buffer_frames: int = 2048
    # Minimum number of unique segments to keep loaded in the pool simultaneously.
    # Enforces shuffle diversity by preventing one large segment from dominating.
    # Violated only when fewer segments are available.
    min_buffer_seg: int = 4
    # Number of decoded chunks the background decode thread can stage in the
    # prefetch queue before it blocks (back-pressure).  Increasing this reduces
    # stall latency spikes at the cost of extra memory:
    #   extra_bytes â‰ˆ prefetch_queue_depth Ã— (buffer_frames/min_buffer_seg) Ã— H Ã— W Ã— 3
    # 0 means "auto" â†’ uses min_buffer_seg.
    prefetch_queue_depth: int = 0
    # Optional frozen recipe manifest; uses cached uint8 clips instead of videos.
    composition_manifest: Optional[str] = None
    composition_original_fraction: float = 0.25
    # MOYO remains available to the normal provider, but not to augmentation.
    composition_datasets: List[str] = field(default_factory=lambda: ["h36m", "mpi"])
    # "prefer" logs missing/known-bad official MPI masks and uses background extraction.
    # Corrupt, misaligned, or incomplete official assets never silently fall back.
    composition_mpi_masks: str = "background"  # background | prefer | required

    # --- Streaming (arbitrarily long sequences) ---
    # Units are windows of seq_len frames: 1 = one window (the minimum), 0 = the
    # whole clip; a clip shorter than a requested length is used whole.  Every
    # pass over a component cuts each of its clips into consecutive pieces at
    # fresh random places (no window is skipped) and shuffles the pieces; a lane
    # walks one piece with the model's memory/track state carried across batches
    # and resets at the next.  The *_window_share lists give the fraction of a
    # component's windows seen in pieces of each length (not draw frequencies).
    # Domain components (stream_domain_components, the long-lived gym/monitoring
    # recordings) use the domain policy; the auxiliary true-3D datasets use the
    # aux policy.  Both [1]/[1.0] = independent windows (streaming off).
    stream_domain_components: List[str] = field(default_factory=lambda: ["gym"])
    stream_domain_chunks: List[int] = field(default_factory=lambda: [1])
    stream_domain_window_share: List[float] = field(default_factory=lambda: [1.0])
    stream_aux_chunks: List[int] = field(default_factory=lambda: [1])
    stream_aux_window_share: List[float] = field(default_factory=lambda: [1.0])
    # Identity gaps of at most this many frames in which no camera observes a
    # person (between two observations of the same id) count as "present but
    # unobserved": the track is kept alive without supervision instead of being
    # trained to die.  Longer absences end the track (re-identification).
    track_max_gap: int = 30
    # Training-only camera dropout (data/augment.py): with this probability a streamed piece's
    # lane drops cameras; every frame then keeps an independent random subset of
    # camera_dropout_min_views..N-1 views (changing within and across windows), down to one camera.
    camera_dropout_prob: float = 0.0
    camera_dropout_min_views: int = 1


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
    # Registered backbone name (models/backbones.py).  When set it overrides
    # yolo_model_name/yolo_feature_indices and verifies the checkpoint hash.
    backbone: Optional[str] = None

    # YOLO-pose backbone (from ultralytics PoseModel, pre-head feature map tap).
    yolo_model_name: str = "yolo11n-pose.pt"
    # Layers to tap for multi-scale features (default: layers 16, 19, 22).
    yolo_feature_indices: List[int] = field(default_factory=lambda: [16, 19, 22])
    yolo_trainable: bool = True
    yolo_force_download: bool = True

    # ViT backbone (registered ``backbone`` of kind "vit": DINOv3 + SimpleFPN, models/vit_backbone.py).
    # Last N ViT blocks (+ final norm) trained; the rest run as a frozen bf16 copy.
    vit_trainable_blocks: int = 0
    # torch.compile the frozen trunk (compute-bound: ~8% on a 4090; first steps pay compilation).
    vit_compile: bool = False
    # SimpleFPN levels: 3 = strides 8/16/32, 4 adds stride 64.  0 = by image size
    # (models/vit_backbone.py: fpn_levels_for): 3 at 640 px (a 10 x 10 stride-64 map is too coarse
    # to help), 4 from 1024 px (e.g. 1280 px: 20 x 20, needed for large people).
    vit_fpn_levels: int = 0

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

    # --- TrackFormer-style track queries (models/tracking.py) ---
    # When enabled, the top-k propagated queries are replaced by max_tracks
    # identity-carrying track slots (num_propagated is ignored); the streaming
    # memory bank remains as attention context.
    track_queries: bool = False
    max_tracks: int = 24
    track_drop_prob: float = 0.1     # training: drop a carried track (re-detection)
    track_fp_prob: float = 0.1       # training: inject a false track per lane-frame
    track_ref_noise: float = 0.005   # training: jitter of carried reference (normalized)
    track_birth_score: float = 0.5   # inference: start / revive a track
    track_keep_score: float = 0.4    # inference: keep an active track
    track_max_inactive: int = 30     # inference: frames an inactive track waits for re-id
    track_nms_mm: float = 300.0      # inference: suppress births near reported people
    # Training assignment entirely on the device (models/gpu_assignment.py): batched matching
    # cost, a GPU Hungarian solver and tensor slot rules -- no per-frame host round trips.
    # Same rules as the host path; random draws (drops, false tracks, ref noise) differ.
    gpu_assignment: bool = False

    # --- Re-identification (models/reid.py; requires track_queries) ---
    # Identity embedding = fusion of a learned projection of the track query
    # (implicit body shape / proportions / motion, averaged by a per-track EMA)
    # and an appearance code sampled from the backbone features at the person's
    # projected keypoints in every view.  Trained with a supervised contrastive
    # loss against a per-lane bank of earlier embeddings (long-range positives).
    reid_dims: int = 0               # 0 disables re-identification
    reid_appearance: bool = True
    reid_ema: float = 0.9            # per-track EMA of identity codes
    reid_loss_weight: float = 1.0
    reid_temperature: float = 0.1
    reid_bank_recent: int = 16       # per identity: recent bank entries ...
    reid_bank_anchors: int = 4       # ... plus the first entries ever seen (long gaps)
    reid_bank_stride: int = 8        # frames between bank insertions per identity
    # Inference: a newly born track is provisional; after reid_wait_frames active
    # frames its EMA embedding is matched against the lane's gallery of ended
    # tracks and, at cosine >= reid_threshold, takes over that identity.
    reid_wait_frames: int = 20
    reid_threshold: float = 0.6
    reid_gallery_size: int = 128
    # Optional frozen pretrained person re-ID network (TorchScript from
    # tools/export_fastreid.py) on per-view crops around each live track's
    # projected joints, every reid_external_stride frames; each track keeps an
    # EMA of its appearance embedding, fused into the identity code.
    reid_external_model: Optional[str] = None
    reid_external_stride: int = 4
    # > 0: run the crop network on exactly this many crops per call (the highest-weight usable
    # views of live tracks), giving static shapes and no device sync; 0: every usable view.
    reid_external_max_crops: int = 0

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
    cost_root: float = 5.0  # used for matching unsupervised frames
    cost_root_labeled: float = 0.0  # optional root tie-breaker on labeled frames
    loss_class: float = 2.0
    loss_kpt: float = 5.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class LossConfig:
    # Root L1 is disabled by default to preserve historical dense-training loss.
    # Sparse datasets should set a positive weight (e.g. 5.0).
    root_weight: float = 0.0
    # --- Approximate ground truth (models/supervision.py) ---
    # Triangulated targets carry a per-joint sigma (mm).  Errors within
    # gt_noise_tolerance * sigma are free, the rest is L1 down-weighted by
    # gt_sigma_ref_mm / (gt_sigma_ref_mm + sigma).  sigma = 0 (exact mocap GT) is plain L1.
    gt_noise_tolerance: float = 1.0
    gt_sigma_ref_mm: float = 30.0
    # People observed in some camera without a 3-D target are supervised by 2-D
    # reprojection: residual beyond reproj_tolerance_px (detector noise) is L1 in
    # units of reproj_scale_px; reproj_triangulated_weight optionally adds the same
    # term for people that do have 3-D targets.
    reproj_weight: float = 1.0
    reproj_tolerance_px: float = 4.0
    reproj_scale_px: float = 64.0
    reproj_triangulated_weight: float = 0.0
    # Track mode: auxiliary decoder layers reuse the final layer's matches (one Hungarian per
    # frame instead of one per layer) and all layers are scored in one vectorized pass.
    share_aux_matching: bool = False


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    max_epochs: int = 20
    # Successful optimizer updates; 0 keeps the existing epoch-only budget.
    max_steps: int = 0
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 35.0
    warmup_iters: int = 100
    log_interval: int = 10
    ckpt_interval: int = 1  # epochs
    # Also overwrite last.pt every N optimizer steps mid-epoch (0 = off), so a
    # killed run loses at most N steps when resumed with --resume.
    ckpt_every_steps: int = 0
    # Cap PyTorch's CUDA cache at this fraction of device memory (0 = off).
    # On Windows the driver otherwise spills an over-grown cache into shared
    # system memory, which silently slows training several-fold.
    cuda_memory_fraction: float = 0.0
    # Queue each step without waiting for the device (engine/step.py: step_deferred): a
    # non-finite loss/gradient skips the update on the device (fused optimizer found_inf), and
    # the step's scalars are read after the next step is queued.  Needs CUDA (fused AdamW);
    # viz steps and other devices use the synchronous step.
    deferred_step: bool = True
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
    # A reported person matches a GT person seen only in 2-D (no 3-D target) when its mean
    # reprojection error over their detected joints is below this (model input pixels).
    reproj_match_px: float = 40.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
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
