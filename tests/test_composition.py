import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from unified_perception.config import Config
from unified_perception.constants import JOINT_NAMES
from unified_perception.data.collate import collate_batch
from unified_perception.data.composition import (
    VERSION, Clip, CompositionDataset, CompositionRejected, compose, file_hash,
    generate_recipes, validate_catalog,
)
from unified_perception.data.dataset import MultiViewPoseDataset
from unified_perception.engine.evaluate import run_eval
from unified_perception.engine.metrics import IdentityAccumulator, MetricsAccumulator
from unified_perception.tools.compose import preview, review
from unified_perception.data.transforms import resize_image_and_intrinsics
from unified_perception.data.backends import HumanPriorAlignedProvider, _split_sequences


def make_clip(identity="a", split="train", offset=0, depth=1500, color=80, cameras=2):
    imgs = np.full((3, cameras, 3, 16, 16), color, dtype=np.uint8)
    masks = np.zeros((3, cameras, 16, 16), dtype=bool)
    masks[..., 4:12, 4:12] = True
    joints = np.zeros((3, 12, 3), dtype=np.float32)
    joints[..., 0] = offset
    joints[..., 2] = depth
    projection = np.broadcast_to(np.eye(4, dtype=np.float32), (3, cameras, 4, 4)).copy()
    metadata = dict(id=identity, split=split, split_group=identity, capture_group="rig",
                    subject="same-actor", sequence=identity, camera_ids=list(range(cameras)),
                    source_sequence=identity, dataset="h36m",
                    frame_ids=[0, 1, 2], fps=10.0, extraction_hash="local-test",
                    geometry_verified=True, distortion="undistorted", approved=True)
    return Clip(metadata, imgs, masks, joints, projection, projection.copy())


BOUNDS = [-6000., -6000., -200., 6000., 6000., 9000.]


@pytest.mark.parametrize("shape,out_hw", [((20, 20, 3), (16, 16)), ((20, 16, 3), (16, 16)),
                                         ((8, 16, 3), (16, 16)), ((8, 8, 3), (16, 16)),
                                         ((21, 20, 3), (16, 16))])
def test_tensor_resize_matches_image_and_camera_contract(shape, out_hw):
    image = torch.full(shape, 80, dtype=torch.uint8)
    resized, intrinsics = resize_image_and_intrinsics(image, np.eye(3), out_hw)
    expected, expected_intrinsics = resize_image_and_intrinsics(image.numpy(), np.eye(3), out_hw)
    assert resized.shape == (*out_hw, 3)
    np.testing.assert_allclose(resized.numpy(), expected, atol=1)
    np.testing.assert_allclose(intrinsics, expected_intrinsics)


@pytest.mark.parametrize("camera_limit,allow_fewer", [(2, False), (7, True)])
def test_fixed_camera_window_returns_raw_pixels(monkeypatch, camera_limit, allow_fewer):
    provider = object.__new__(HumanPriorAlignedProvider)
    provider.cfg = Config()
    provider.cfg.data.allow_fewer_cameras = allow_fewer
    provider._seqs = {"seq": SimpleNamespace(num_frames=3)}
    provider.seq_len, provider.max_people, provider.num_cameras = 3, 8, camera_limit
    provider.out_hw = (16, 16)
    provider.mean, provider.std = [0, 0, 0], [255, 255, 255]
    monkeypatch.setattr(provider, "_dataset_cfg_for_seq", lambda seq: None)
    frames = [dict.fromkeys(JOINT_NAMES, [0, 0, 1500])] * 3
    monkeypatch.setattr(provider, "_mocap_frames", lambda seq: frames)
    definitions = {c: {"K": np.eye(3), "R": np.eye(3), "T": np.zeros(3)} for c in ("a", "b")}
    monkeypatch.setattr(provider, "_cam_defs_for_seq", lambda seq: definitions)
    def images(ds, sample, cameras, count):
        return cameras, {c: [torch.full((20, 20, 3), 80, dtype=torch.uint8)] * 3 for c in cameras}
    monkeypatch.setattr(provider, "_load_window_images", images)
    monkeypatch.setattr(provider, "_desired_frame_indices", lambda *args: [0, 1, 2])
    raw = provider.load_window(("seq", 0), camera_ids=["b", "a"])
    assert raw["camera_ids"] == ["b", "a"]
    assert raw["imgs"].dtype == torch.uint8
    assert raw["imgs"][0, 0, 0, 8, 8] == 80


def test_video_frame_mapping_respects_traceability_and_camera_offset():
    provider = object.__new__(HumanPriorAlignedProvider)
    provider.seq_len = 3
    provider._seqs = {"seq": SimpleNamespace(mocap_indices=[9, 17, 25], stride=8)}
    dataset = SimpleNamespace(mocap_fps=60, camera_fps=30, get_camera_image_offset=lambda camera: 4)
    assert provider._desired_frame_indices(dataset, ("seq", 0), "camera") == [0, 4, 8]


def test_raw_test_split_does_not_silently_load_training():
    with pytest.raises(ValueError, match="explicit"):
        _split_sequences(["a", "b"], "test", .2, 0)


def write_cache(root, clip):
    entry = clip.metadata.copy()
    path = root / (entry["id"] + ".npz")
    np.savez_compressed(path, imgs=clip.imgs, masks=clip.masks, joints=clip.joints,
                        world2img=clip.world2img, intrinsics=clip.intrinsics)
    entry.update(path=path.name, sha256=file_hash(path))
    return entry


def manifest_for(root):
    entries = [write_cache(root, make_clip("a", depth=1500)),
               write_cache(root, make_clip("b", depth=4000, color=180))]
    clips = [make_clip("a", depth=1500), make_clip("b", depth=4000, color=180)]
    parameters = dict(position_range=BOUNDS, radius_mm=150., depth_margin_mm=150.)
    _, _, order = compose(clips, **parameters)
    recipe = dict(id="train-0000", split="train", sources=["a", "b"], approved=True,
                  order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
    manifest = dict(version=VERSION, clips=entries, recipes=[recipe],
                    parameters=parameters, quality_approved=True)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    cfg = Config()
    cfg.data.composition_manifest = str(path)
    cfg.data.seq_len, cfg.data.num_cameras = 3, 2
    cfg.data.image_height = cfg.data.image_width = 16
    cfg.data.max_people = 8
    cfg.model.position_range = BOUNDS
    return cfg, manifest, path


def test_foreground_base_is_protected_and_hidden_targets_remain(tmp_path):
    cfg, _, _ = manifest_for(tmp_path)
    sample = MultiViewPoseDataset(cfg)[0]
    assert sample["imgs"].dtype == torch.uint8
    assert sample["imgs"][0, 0, 0, 6, 6] == 80
    assert sample["gt_valid"][:, :2].all()
    assert (sample["visibility"][:, :, 1] == 0).all()
    assert (sample["instance_ids"][:, :2] == torch.tensor([0, 1])).all()
    assert not sample["gt_valid"][:, 2:].any()
    batch = collate_batch([sample, sample])
    assert batch["instance_ids"].shape == (2, 3, 8)
    assert batch["visibility"].shape == (2, 3, 2, 8)


def test_order_changes_per_camera():
    a, b = make_clip(depth=1500), make_clip("b", depth=4000, color=180)
    # Camera 1 looks in the opposite direction from z=6000.
    for clip in (a, b):
        clip.world2img[:, 1, 0, 0] = -1
        clip.world2img[:, 1, 2, 2] = -1
        clip.world2img[:, 1, 2, 3] = 6000
    imgs, visible, order = compose([a, b], BOUNDS)
    assert imgs[0, 0, 0, 6, 6] == 80
    assert imgs[0, 1, 0, 6, 6] == 180
    np.testing.assert_array_equal(order[0], [[0, 1], [1, 0]])
    assert visible[0, 0, 1] == visible[0, 1, 0] == 0


def test_overlap_blending_is_local_reproducible_and_label_preserving():
    near, far = make_clip(), make_clip("b", depth=4000, color=180)
    near.masks[:] = False
    far.masks[:] = False
    near.masks[..., 4:12, 4:10] = True
    far.masks[..., 4:12, 8:14] = True
    original_masks = [c.masks.copy() for c in (near, far)]
    original_images = [c.imgs.copy() for c in (near, far)]
    hard, visibility, order = compose([near, far], BOUNDS, blend_intersections=False)
    mixed, mixed_visibility, mixed_order = compose([near, far], BOUNDS)
    np.testing.assert_array_equal(mixed_visibility, visibility)
    np.testing.assert_array_equal(mixed_order, order)
    np.testing.assert_array_equal(mixed, compose([near, far], BOUNDS)[0])
    changed = np.any(mixed != hard, axis=2)
    expected = np.zeros_like(changed)
    expected[..., 4:12, 9:11] = True
    np.testing.assert_array_equal(changed, expected)
    assert ((mixed[:, :, :, 4:12, 9:11] > 80) & (mixed[:, :, :, 4:12, 9:11] < 180)).all()
    assert np.unique(mixed[0, 0, 0, 4:12, 9:11]).size > 2
    assert not np.array_equal(mixed[0], mixed[1])
    for clip, masks, images in zip((near, far), original_masks, original_images):
        np.testing.assert_array_equal(clip.masks, masks)
        np.testing.assert_array_equal(clip.imgs, images)


@pytest.mark.parametrize("far_start", [8, 10])
def test_nonoverlapping_or_touching_people_are_not_blended(far_start):
    near, far = make_clip(), make_clip("b", depth=4000, color=180)
    near.masks[:] = False
    far.masks[:] = False
    near.masks[..., 4:12, 4:8] = True
    far.masks[..., 4:12, far_start:14] = True
    np.testing.assert_array_equal(
        compose([near, far], BOUNDS)[0],
        compose([near, far], BOUNDS, blend_intersections=False)[0],
    )


def test_hidden_people_and_originals_do_not_create_blending_edges():
    clips = [make_clip(), make_clip("b", depth=4000, color=180),
             make_clip("c", depth=6500, color=240)]
    np.testing.assert_array_equal(compose(clips, BOUNDS)[0], clips[0].imgs)
    np.testing.assert_array_equal(compose(clips[:1], BOUNDS)[0], clips[0].imgs)


def test_blending_handles_image_edges_without_wrapping():
    near, far = make_clip(), make_clip("b", depth=4000, color=180)
    near.masks[:] = False
    far.masks[:] = True
    near.masks[..., :8, :8] = True
    mixed = compose([near, far], BOUNDS)[0]
    assert mixed.dtype == np.uint8
    assert (mixed[..., 0, 0] == 80).all()
    assert (mixed[..., -1, -1] == 180).all()
    assert (mixed[..., 7, 7] > 80).all()


def test_old_compositor_approvals_are_not_reused(tmp_path):
    cfg, manifest, path = manifest_for(tmp_path)
    manifest["version"] = VERSION - 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        CompositionDataset(cfg, "train")
    with pytest.raises(ValueError, match="Regenerate"):
        preview(SimpleNamespace(manifest=str(path), output=str(tmp_path / "preview"), ids=None, limit=1))
    with pytest.raises(ValueError, match="Regenerate"):
        review(SimpleNamespace(manifest=str(path), review=str(tmp_path / "unused.json")))


def test_collision_and_crossing_trajectories_rejected():
    a, b = make_clip(), make_clip("b")
    with pytest.raises(CompositionRejected, match="collision"):
        compose([a, b], BOUNDS)
    a.joints[..., 0] = np.array([-1000, 1000, 1000])[:, None]
    b.joints[..., 0] = np.array([1000, -1000, -1000])[:, None]
    with pytest.raises(CompositionRejected, match="collision"):
        compose([a, b], BOUNDS)


def test_ambiguous_overlap_is_not_pelvis_sorted():
    a, b = make_clip(offset=-500), make_clip("b", offset=500, depth=1600)
    with pytest.raises(CompositionRejected, match="ambiguous"):
        compose([a, b], BOUNDS)


@pytest.mark.parametrize("field,value", [("split", "test"), ("capture_group", "other"),
                                        ("camera_ids", [1, 0]), ("fps", 20)])
def test_incompatible_sources_rejected(field, value):
    a, b = make_clip(), make_clip("b", depth=4000)
    b.metadata[field] = value
    with pytest.raises(CompositionRejected, match="incompatible"):
        compose([a, b], BOUNDS)


def test_shifted_calibration_rejected():
    a, b = make_clip(), make_clip("b", depth=4000)
    b.world2img[..., 0, 3] = 1
    with pytest.raises(CompositionRejected, match="calibration"):
        compose([a, b], BOUNDS)


def test_split_group_and_sequence_leakage():
    entries = [make_clip().metadata, make_clip("b", split="test").metadata]
    entries[1]["split_group"] = "a"
    with pytest.raises(ValueError, match="leakage"):
        validate_catalog(entries)


def test_moyo_cache_is_excluded_from_augmentation(tmp_path):
    cfg, manifest, path = manifest_for(tmp_path)
    manifest["clips"][0]["dataset"] = "moyo"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="excluded"):
        CompositionDataset(cfg, "train")


def test_generation_reproducible_and_bounded(tmp_path):
    entries = [write_cache(tmp_path, make_clip("a", depth=1500)),
               write_cache(tmp_path, make_clip("b", depth=4000))]
    args = (entries, tmp_path, "train", 2, [2], 21, BOUNDS)
    recipes, report = generate_recipes(*args, max_attempts=20)
    assert report["achieved"] == 2
    assert (recipes, report) == generate_recipes(*args, max_attempts=20)
    recipes, report = generate_recipes(entries, tmp_path, "train", 3, [2, 3], 21, BOUNDS, max_attempts=20)
    assert report["achieved"] < report["requested"]
    assert report["rejections"]["insufficient_sources_3"] == 1


def test_tampering_and_unreviewed_manifest_fail(tmp_path):
    cfg, manifest, path = manifest_for(tmp_path)
    manifest["quality_approved"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="review"):
        CompositionDataset(cfg, "train")
    manifest["quality_approved"] = True
    path.write_text(json.dumps(manifest))
    asset = tmp_path / "a.npz"
    asset.write_bytes(asset.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Stale"):
        CompositionDataset(cfg, "train")


def test_capacity_rejected_and_original_control(tmp_path):
    cfg, _, _ = manifest_for(tmp_path)
    cfg.data.max_people = 1
    with pytest.raises(ValueError, match="capacity"):
        CompositionDataset(cfg, "train")
    cfg.data.max_people = 8
    cfg.data.composition_original_fraction = 1
    dataset = CompositionDataset(cfg, "train")
    assert len(dataset) == 2
    assert dataset[0]["gt_valid"].sum() == 3


def test_empty_matches_are_not_perfect_pose_scores():
    accumulator = MetricsAccumulator()
    assert accumulator.update_frame(np.zeros((0, 12, 3)), np.zeros((2, 12, 3))) == []
    metrics = accumulator.compute()
    assert np.isnan(metrics["mpjpe_mm"])
    assert metrics["false_negatives"] == 2
    assert metrics["matched_coverage"] == 0


def test_identity_switches_gaps_and_reset():
    accumulator = IdentityAccumulator()
    accumulator.update(np.array([7]), np.array([0]), [(0, 0)])
    accumulator.update(np.array([]), np.array([0]), [])
    accumulator.update(np.array([8]), np.array([0]), [(0, 0)])
    accumulator.reset_clip()
    accumulator.update(np.array([2]), np.array([0]), [(0, 0)])
    metrics = accumulator.compute()
    assert metrics["id_switches"] == 1
    assert metrics["id_fragments"] == 3
    assert metrics["id_matched_coverage"] == .75


def test_review_excludes_failures_and_enforces_pilot_gate(tmp_path):
    _, manifest, path = manifest_for(tmp_path)
    decisions = tmp_path / "review.json"
    decisions.write_text(json.dumps(dict(reviewer="test", rubric="mask and temporal check",
                                         approved=[], rejected=["train-0000"])))
    with pytest.raises(ValueError, match="95%"):
        review(SimpleNamespace(manifest=str(path), review=str(decisions)))
    assert json.loads(path.read_text())["quality_approved"] is False


def test_explicit_review_rollout_requires_representative_strata(tmp_path):
    _, manifest, path = manifest_for(tmp_path)
    manifest["recipes"].append(dict(manifest["recipes"][0], id="train-0001"))
    path.write_text(json.dumps(manifest))
    decisions = tmp_path / "review.json"
    decisions.write_text(json.dumps(dict(reviewer="test", rubric="representative temporal review",
                                         approved=["train-0000"], rejected=[], approve_remaining=True)))
    review(SimpleNamespace(manifest=str(path), review=str(decisions)))
    document = json.loads(path.read_text())
    assert all(r["approved"] for r in document["recipes"])
    document["recipes"].append(dict(document["recipes"][0], id="train-0002", sources=["a", "b", "a"]))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="stratum"):
        review(SimpleNamespace(manifest=str(path), review=str(decisions)))


def test_preview_source_and_composite(tmp_path):
    _, manifest, path = manifest_for(tmp_path)
    output = tmp_path / "preview"
    preview(SimpleNamespace(manifest=str(path), output=str(output), ids=None, limit=1))
    assert (output / "train-0000.avi").stat().st_size > 0
    del manifest["recipes"]
    path.write_text(json.dumps(manifest))
    preview(SimpleNamespace(manifest=str(path), output=str(output), ids=["a"], limit=1))
    assert (output / "a.avi").stat().st_size > 0


def test_evaluation_metadata_and_hidden_target_recall(tmp_path):
    cfg, _, _ = manifest_for(tmp_path)
    loader = DataLoader(MultiViewPoseDataset(cfg), batch_size=1, collate_fn=collate_batch)

    class FixtureModel(torch.nn.Module):
        def forward(self, imgs, projection, intrinsics, camera_valid=None):
            b, t = imgs.shape[:2]
            cls = torch.full((1, b, t, 8, 1), -20.)
            cls[..., 0, :] = 20.
            kpts = torch.zeros((1, b, t, 8, 12, 3))
            kpts[..., 0, :, 2] = 1500.
            return cls, kpts, None, None

        def denormalize_keypoints(self, keypoints):
            return keypoints

    metrics = run_eval(FixtureModel(), loader, cfg, torch.device("cpu"))
    assert metrics["recall"] == .5
    assert metrics["visibility_hidden/recall"] == 0
    assert metrics["id_switches"] == 0
    assert metrics["id_matched_coverage"] == .5


def test_composed_batch_through_model_loss_and_optimizer(tmp_path, monkeypatch):
    from unified_perception.models import detector
    from unified_perception.models.losses import SetCriterion

    class TinyBackbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.layer = torch.nn.Conv2d(3, 24, kernel_size=3, stride=4, padding=1)

        def forward(self, images, return_detections=False):
            return [self.layer(images)], None

    monkeypatch.setattr(detector, "YOLOPoseBackbone", TinyBackbone)
    cfg, _, _ = manifest_for(tmp_path)
    cfg.model.embed_dims = 24
    cfg.model.dec_layers = 2
    cfg.model.dec_heads = 2
    cfg.model.dec_ffn_dim = 32
    cfg.model.num_queries = 8
    cfg.model.num_propagated = 2
    cfg.model.memory_len = 8
    cfg.model.memory_topk = 8
    cfg.model.dec_num_pts = 1
    cfg.model.dn_scalar = 2
    cfg.model.use_mmcv_msda = False
    batch = collate_batch([MultiViewPoseDataset(cfg)[0]])
    model = detector.UnifiedPerceptionModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cls, kpts, dn, _ = model(
        batch["imgs"].float() / 255, batch["world2img"], batch["intrinsics"],
        model.normalize_keypoints(batch["gt_keypoints"]), batch["gt_valid"],
    )
    loss, _ = SetCriterion(cfg.model)(
        cls, kpts, model.normalize_keypoints(batch["gt_keypoints"]), batch["gt_valid"], dn,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()


def test_worker_count_does_not_change_cached_composites(tmp_path):
    cfg, _, _ = manifest_for(tmp_path)
    dataset = MultiViewPoseDataset(cfg)
    sequential = list(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_batch))
    parallel = list(DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=collate_batch))
    assert len(sequential) == len(parallel)
    for left, right in zip(sequential, parallel):
        torch.testing.assert_close(left["imgs"], right["imgs"])
        torch.testing.assert_close(left["gt_keypoints"], right["gt_keypoints"])
