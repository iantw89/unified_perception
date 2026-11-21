"""Gym import/provider/mixture tests; generated fixtures stay under the ignored project cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
import pytest
import torch

from unified_perception.config import Config, load_config
from unified_perception.constants import JOINT_NAMES
from unified_perception.data import MultiViewPoseDataset, collate_batch
from unified_perception.data.composition import VERSION, Clip, compose, file_hash
from unified_perception.data.gym_pseudo3d import (
    GymMocapMixDataset, GymPseudo3DDataset, GymPseudo3DProvider, labeled_window_starts,
)
from unified_perception.engine.evaluate import run_eval
from unified_perception.engine.train import build_loaders
from unified_perception.tools.import_gym_pseudo3d import (
    CAMERA_IDS, convert, observation_masks, transform_targets,
)

REPO = Path(__file__).resolve().parents[1]
WIDE_RANGE = [-6000., -12000., -200., 6000., 12000., 3000.]


@pytest.fixture
def workspace():
    path = REPO / "data_cache" / ("_gym_test_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def source_arrays():
    frames, people = 301, 2
    root = np.broadcast_to(np.array([1000., 2000., 1000.]), (frames, people, 3)).copy()
    root[:, 1, 0] += 300
    exists = np.ones((frames, people), dtype=bool)
    exists[6:16, 1] = False  # absent at label 10; nearest-label propagation (ties to earlier)
    labeled = np.arange(frames) % 10 == 0
    joint_valid = np.broadcast_to(labeled[:, None, None], (frames, people, 12)).copy()
    joint_valid[:, 1] = False  # present with rough root, but no observed joints
    return dict(frames=np.arange(frames), person_ids=np.array([9, 21]), exists=exists,
                root=root, root_valid=exists.copy(), root_source=np.ones((frames, people), np.int8),
                keypoints=np.broadcast_to(root[:, :, None], (frames, people, 12, 3)).copy(),
                keypoint_valid=joint_valid, labeled=labeled, joint_names=np.array(JOINT_NAMES))


def source_observations(source):
    """Person 9: cameras 0 and 1. Person 21: camera 6 only."""
    identity = {"0:1": 9, "1:1": 9, "6:2": 21}
    observations = []
    for frame in np.flatnonzero(source["labeled"]):
        for key, person in identity.items():
            if source["exists"][frame, [9, 21].index(person)]:
                camera, track = map(int, key.split(":"))
                xy = [-500., -500.] if (person, frame) == (21, 200) else [640., 480.]  # 200: outside crop
                observations.append(dict(camera=camera, track=track, session_frame=int(frame),
                                         joints={"left_hip": {"xy": xy, "occluded": False}}))
    return observations, identity


@pytest.fixture
def converted(workspace, monkeypatch):
    source = source_arrays()
    np.savez(workspace / "source.npz", **source)
    observations, identity = source_observations(source)
    (workspace / "observations.json").write_text(json.dumps(observations), encoding="utf-8")
    (workspace / "identity.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
    K = np.array([[800., 0, 640], [0, 800., 480], [0, 0, 1]])
    definition = dict(K=K.tolist(), D=[0.] * 14, R=np.eye(3).tolist(), T_mm=[-1150., -2000., 4000.])
    (workspace / "cameras.json").write_text(json.dumps({"cameras": {str(c): definition for c in CAMERA_IDS}}))

    class Capture:
        def __init__(self, path):
            self.frame = np.zeros((960, 1280, 3), np.uint8)
            self.frame[..., 2] = 100
        def isOpened(self):
            return True
        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 960, cv2.CAP_PROP_FPS: 10}[prop]
        def read(self):
            return True, self.frame
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    root = workspace / "cache"
    index = convert(workspace / "source.npz", workspace / "cameras.json", workspace, root, (32, 32),
                    observations=workspace / "observations.json", identity=workspace / "identity.json")
    return root, source, index


def cache_config(root, cameras=4):
    cfg = Config()
    cfg.data.provider = "gym_pseudo3d"
    cfg.data.gym_pseudo3d_root = str(root)
    cfg.data.num_cameras, cfg.data.max_people, cfg.data.seq_len = cameras, 3, 8
    cfg.data.image_height = cfg.data.image_width = 32
    cfg.data.use_shuffle_buffer = False
    cfg.data.num_workers = 0
    cfg.model.position_range = list(WIDE_RANGE)
    return cfg


def test_transform_masks_train_only_origin_and_joint_order():
    source = source_arrays()
    source["root"][240:, :, 0] += 100000
    arrays, offset = transform_targets(source)
    np.testing.assert_allclose(offset, [1150, 2000, 0])
    assert arrays["valid"][0, 1] and arrays["root_valid"][0, 1]
    assert not arrays["keypoint_valid"][0, 1].any()
    assert not arrays["keypoint_valid"][~arrays["labeled"]].any()
    assert (arrays["keypoints"][~arrays["keypoint_valid"]] == 0).all()
    source["joint_names"] = source["joint_names"][::-1]
    with pytest.raises(ValueError, match="joint_names"):
        transform_targets(source)


def test_observation_masks_use_nearest_label_and_must_match_exists():
    source = source_arrays()
    observations, identity = source_observations(source)
    observed = observation_masks(observations, identity, source["person_ids"], source["labeled"], source["exists"])
    assert observed.shape == (301, 2, 4)
    assert observed[5, 1, 3] and not observed[6, 1].any() and not observed[15, 1].any() and observed[16, 1, 3]
    assert observed[:, 0, :2].all() and not observed[:, 0, 2:].any()
    cropped = observation_masks(observations, identity, source["person_ids"], source["labeled"],
                                source["exists"], in_view=lambda item: item["joints"]["left_hip"]["xy"][0] > 0)
    assert not cropped[196:206, 1].any() and cropped[195, 1, 3] and cropped[206, 1, 3]
    np.testing.assert_array_equal(np.delete(cropped, np.s_[196:206], 0), np.delete(observed, np.s_[196:206], 0))
    exists = source["exists"].copy()
    exists[5, 1] = False
    with pytest.raises(ValueError, match="disagree"):
        observation_masks(observations, identity, source["person_ids"], source["labeled"], exists)
    with pytest.raises(ValueError, match="identity"):
        observation_masks(observations, {}, source["person_ids"], source["labeled"], source["exists"])


@pytest.mark.parametrize("seq_len", [1, 8, 10, 16, 300])
def test_window_boundaries_and_label_endpoints(seq_len):
    for first, last in ((0, 239), (240, 300)):
        frames = np.arange(first, last + 1)
        labeled = frames % 10 == 0
        starts = labeled_window_starts(labeled, seq_len)
        for start in starts:
            assert 0 <= start <= len(frames) - seq_len
            assert labeled[start:start + seq_len].any()
            assert start in (0, len(frames) - seq_len) or labeled[start + seq_len - 1]
        if seq_len == 16:
            covered = np.zeros(len(frames), bool)
            for start in starts:
                covered[start:start + seq_len] = True
            assert covered.all()  # every cached count/root receives supervision
        anywhere = labeled_window_starts(labeled, seq_len, "any")
        assert set(starts) <= set(anywhere)
        assert all(labeled[s:s + seq_len].any() and s + seq_len <= len(frames) for s in anywhere)
        assert len(anywhere) == sum(labeled[s:s + seq_len].any() for s in range(len(frames) - seq_len + 1))
    assert labeled_window_starts(np.zeros(30, bool), seq_len) == []
    assert labeled_window_starts(np.zeros(30, bool), seq_len, "any") == []


def test_converter_shapes_split_and_projection(converted):
    root, source, index = converted
    assert index["camera_ids"] == [0, 1, 2, 6] and index["format_version"] == 2
    assert index["dropped_crop_person_frames"] == 10
    for split, first, stop in (("train", 0, 240), ("val", 240, 301)):
        folder = root / split / "gym_20250505"
        poses, calib = np.load(folder / "poses.npz"), np.load(folder / "calib.npz")
        np.testing.assert_array_equal(poses["frames"], np.arange(first, stop))
        exists = source["exists"].copy()
        exists[196:206, 1] = False  # person 21's only view at label 200 is outside the undistortion crop
        np.testing.assert_array_equal(poses["valid"], exists[first:stop])
        assert not poses["root_valid"][~poses["valid"]].any()
        np.testing.assert_array_equal(poses["observed"].any(-1), poses["valid"])
        assert poses["observed"].shape == (stop-first, 2, 4)
        assert poses["keypoints"].shape == (stop-first, 2, 12, 3)
        assert poses["keypoint_valid"].shape == (stop-first, 2, 12)
        assert poses["root"].shape == (stop-first, 2, 3)
        assert len(list((folder / "images").glob("*.png"))) == 4 * (stop-first)
        assert calib["K"].shape == calib["R"].shape == (4, 3, 3)
        assert calib["T"].shape == (4, 3)
        np.testing.assert_allclose(calib["T"][0], [-1150, -2000, 4000] + np.array(index["world_offset_mm"]), atol=1e-5)
        p = poses["root"][0, 0]
        projected = calib["K"][0] @ (calib["R"][0] @ p + calib["T"][0])
        uv = projected[:2] / projected[2]
        assert (uv > 0).all() and (uv < 32).all()
    assert all(Path(path).is_file() for path in index["debug_images"])
    with pytest.raises(FileExistsError):
        convert(root, root, root, root)


def test_dataset_sparse_targets_count_and_calibrated_rgb(converted):
    root, source, _ = converted
    cfg = cache_config(root)
    dataset = MultiViewPoseDataset(cfg, "train")
    # Training uses every in-split 8-frame window containing a label (8 per decade).
    assert len(dataset) == 185
    sample = dataset[0]
    assert sample["imgs"].shape == (8, 4, 3, 32, 32)
    assert sample["imgs"].dtype == torch.uint8
    assert sample["imgs"][0, 0, 0, 16, 16] == 100  # BGR -> RGB
    assert sample["gt_valid"].shape == (8, 3)
    assert sample["gt_keypoints"].shape == (8, 3, 12, 3)
    assert sample["gt_root"].shape == (8, 3, 3)
    assert sample["gt_root_valid"].shape == (8, 3)
    assert sample["gt_keypoint_frame_valid"].tolist() == [True] + [False] * 7
    np.testing.assert_array_equal(sample["gt_valid"][:, :2], source["exists"][:8])
    assert sample["gt_root_valid"][0, 1] and not sample["gt_keypoint_valid"][:, 1].any()
    assert not sample["gt_valid"][:, 2].any()
    assert torch.isfinite(sample["gt_keypoints"]).all()
    projected = sample["world2img"][0, 0] @ torch.cat((sample["gt_root"][0, 0], torch.ones(1)))
    uv = projected[:2] / projected[2]
    assert (uv > 0).all() and (uv < 32).all()
    batch = collate_batch([sample, dataset[1]])
    assert batch["gt_keypoint_valid"].shape == (2, 8, 3, 12)
    val = MultiViewPoseDataset(cfg, "val")
    assert len(val) == 7  # streaming windows end on a label, never cross the split
    assert val[0]["frame_ids"][0] == 240
    assert val[-1]["frame_ids"][-1] == 300
    with pytest.raises(ValueError, match="boundary"):
        val.provider.load_window(("gym_20250505", 60))
    cfg.data.max_people = 1
    with pytest.raises(ValueError, match="truncate"):
        GymPseudo3DProvider(cfg)
    cfg.data.max_people, cfg.data.num_cameras = 3, 5
    with pytest.raises(ValueError, match="reviewed views"):
        GymPseudo3DProvider(cfg)


def test_three_view_subsets_remove_unobserved_people(converted):
    root, source, _ = converted
    cfg = cache_config(root, cameras=3)
    provider = GymPseudo3DProvider(cfg, "train")
    assert len(provider.samples) == 185 * 4  # every 3-of-4 subset of each window
    without_six = provider.load_window(("gym_20250505", 0, (0, 1, 2)))
    assert without_six["imgs"].shape == (8, 3, 3, 32, 32)
    assert without_six["gt_valid"][:, 0].all() and not without_six["gt_valid"][:, 1:].any()
    # Person 21 (seen only by camera 6) keeps its identity slot for track queries, but is
    # unsupervised: present in the scene yet unobserved by the chosen views.
    assert without_six["instance_ids"][0].tolist() == [9, 21, -1]
    # Existence also bridges short all-camera gaps between sightings (data.track_max_gap).
    from unified_perception.data.identity import fill_identity_gaps
    filled = fill_identity_gaps(source["exists"], cfg.data.track_max_gap)
    np.testing.assert_array_equal(without_six["instance_exists"][:, 1], filled[:8, 1])
    assert (filled[:8, 1] & ~source["exists"][:8, 1]).any()  # this fixture has such a gap
    assert not without_six["instance_exists"][:, 2].any()
    assert not without_six["gt_root_valid"][:, 1:].any() and not without_six["gt_keypoint_valid"][:, 1:].any()
    np.testing.assert_array_equal(without_six["visibility"][:, :, 0], [[1, 1, 0]] * 8)
    only_six = provider.load_window(("gym_20250505", 0, (1, 2, 3)))
    # Person 9 is still seen by camera 1; person 21 only by camera 6 (slot 3).
    assert only_six["instance_ids"][0].tolist() == [9, 21, -1]
    np.testing.assert_array_equal(only_six["gt_valid"][:, 1], source["exists"][:8, 1])
    calib = np.load(root / "train" / "gym_20250505" / "calib.npz")
    np.testing.assert_allclose(only_six["intrinsics"][0, 2, :3, :3], calib["K"][3], rtol=1e-6)
    with pytest.raises(ValueError, match="subset"):
        provider.load_window(("gym_20250505", 0, (0, 1)))


def test_six_camera_import_and_seven_camera_ceiling(converted):
    root, _, _ = converted
    workspace = root.parent
    definitions = json.loads((workspace / "cameras.json").read_text())
    for camera in ("4", "5"):
        definitions["cameras"][camera] = definitions["cameras"]["0"]
    (workspace / "six_cameras.json").write_text(json.dumps(definitions))
    observations = json.loads((workspace / "observations.json").read_text())
    observations += [dict(item, camera=camera)
                     for camera in (4, 5) for item in list(observations) if item["camera"] == 0]
    identity = json.loads((workspace / "identity.json").read_text())
    identity["identity"].update({"4:1": 9, "5:1": 9})
    (workspace / "six_observations.json").write_text(json.dumps(observations))
    (workspace / "six_identity.json").write_text(json.dumps(identity))
    cameras = (0, 1, 2, 4, 5, 6)
    target = workspace / "six_cache"
    index = convert(workspace / "source.npz", workspace / "six_cameras.json", workspace,
                    target, (32, 32), observations=workspace / "six_observations.json",
                    identity=workspace / "six_identity.json", camera_ids=cameras)
    assert index["camera_ids"] == list(cameras)
    cfg = cache_config(target, cameras=7)
    cfg.data.allow_fewer_cameras = True
    sample = GymPseudo3DDataset(cfg, "val")[0]
    assert sample["imgs"].shape == (8, 6, 3, 32, 32)
    assert sample["camera_ids"] == list(cameras)
    assert sample["seq"].endswith("views012456")
    assert sample["visibility"].shape == (8, 6, 3)
    assert sample["gt_valid"][:, :2].any()


def test_camera_ceiling_reuses_original_composition_and_pads_batching(converted):
    root, _, _ = converted
    cfg = cache_config(root, cameras=7)
    cfg.data.allow_fewer_cameras = True
    cfg.data.composition_manifest = str(_manifest(root.parent))
    cfg.data.image_height = cfg.data.image_width = 32
    cfg.model.position_range[5] = 9000.0
    sample = MultiViewPoseDataset(cfg, "val")[0]
    assert sample["imgs"].shape == (8, 3, 3, 32, 32)
    gym = GymPseudo3DDataset(cfg, "val")[0]
    assert gym["imgs"].shape[1] == 4
    batch = collate_batch([sample, gym])
    assert batch["imgs"].shape == (2, 8, 4, 3, 32, 32)
    assert batch["camera_valid"][0].sum() == 8 * 3
    assert batch["camera_valid"][1].all()
    assert not batch["visibility"][0, :, 3].any()
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    assert build_loaders(cfg)["train"].batch_size == 2


def _clip(identity, split, depth):
    imgs = np.full((8, 3, 3, 32, 32), 80, dtype=np.uint8)
    masks = np.zeros((8, 3, 32, 32), dtype=bool)
    masks[..., 8:24, 8:24] = True
    joints = np.zeros((8, 12, 3), dtype=np.float32)
    joints[..., 2] = depth
    projection = np.broadcast_to(np.eye(4, dtype=np.float32), (8, 3, 4, 4)).copy()
    metadata = dict(id=identity, split=split, split_group=identity, capture_group="rig",
                    subject=identity, sequence=identity, camera_ids=[0, 1, 2],
                    source_sequence=identity, dataset="h36m", frame_ids=list(range(8)), fps=10.0,
                    extraction_hash="local-test", geometry_verified=True,
                    distortion="undistorted", approved=True)
    return Clip(metadata, imgs, masks, joints, projection, projection.copy())


def _manifest(folder):
    parameters = dict(position_range=[-6000., -6000., -200., 6000., 6000., 9000.],
                      radius_mm=150., depth_margin_mm=150.)
    entries, recipes = [], []
    for split, ids in (("train", ("a", "b")), ("val", ("c", "d"))):
        clips = [_clip(i, split, depth) for i, depth in zip(ids, (1500, 4000))]
        for clip in clips:
            entry = clip.metadata.copy()
            path = folder / f"{entry['id']}.npz"
            np.savez_compressed(path, imgs=clip.imgs, masks=clip.masks, joints=clip.joints,
                                world2img=clip.world2img, intrinsics=clip.intrinsics)
            entry.update(path=path.name, sha256=file_hash(path))
            entries.append(entry)
        _, _, order = compose(clips, **parameters)
        recipes.append(dict(id=f"{split}-0", split=split, sources=list(ids), approved=True,
                            order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
    path = folder / "manifest.json"
    path.write_text(json.dumps(dict(version=VERSION, clips=entries, recipes=recipes,
                                    parameters=parameters, quality_approved=True)), encoding="utf-8")
    return path


class _ConstantModel(torch.nn.Module):
    """Predicts every query at the gym fixture's hip centre."""

    def forward(self, imgs, world2img, intrinsics, camera_valid=None):
        b, t = imgs.shape[:2]
        scores = torch.zeros((1, b, t, 4, 1))
        kpts = torch.zeros((1, b, t, 4, 12, 3))
        return scores, kpts, None, None

    def denormalize_keypoints(self, kpts):
        return kpts + torch.tensor([-150., 0., 1000.])


def test_mix_build_loaders_real_train_and_val_paths(converted):
    root, _, _ = converted
    folder = root.parent / "composition"
    folder.mkdir()
    cfg = cache_config(root, cameras=3)
    cfg.data.provider = "gym_mocap_mix"
    cfg.data.composition_manifest = str(_manifest(folder))
    cfg.data.composition_original_fraction = 0.25
    cfg.data.gym_sample_fraction = 0.25
    cfg.model.position_range = [-6000., -12000., -200., 6000., 12000., 9000.]  # contains both
    cfg.data.batch_size = 4
    cfg.model.num_joints = 12
    loaders = build_loaders(cfg)
    assert set(loaders) == {"train", "val", "val_gym"}
    train = loaders["train"].dataset
    assert isinstance(train, GymMocapMixDataset)
    assert len(train.mocap) == 1 + round(1 * .25 / .75) and len(train.gym) == 1
    assert isinstance(loaders["val_gym"].dataset, GymPseudo3DDataset)
    assert len(loaders["val_gym"].dataset) == 7 * 4
    mixed = collate_batch([train[0], train[len(train) - 1]])  # composition + gym in one batch
    assert mixed["imgs"].shape == (2, 8, 3, 3, 32, 32)
    assert mixed["visibility"].shape == (2, 8, 3, 3) and mixed["instance_ids"].shape == (2, 8, 3)
    for name in ("val", "val_gym"):
        batch = next(iter(loaders[name]))
        assert batch["imgs"].shape[1:3] == (8, 3)
        metrics = run_eval(_ConstantModel(), loaders[name], cfg, torch.device("cpu"))
        assert metrics["target_persons"] > 0
    gym = run_eval(_ConstantModel(), loaders["val_gym"], cfg, torch.device("cpu"))
    assert gym["matched_persons"] > 0 and np.isfinite(gym["mpjpe_mm"])
    assert "capture_gym_20250505/f1" in gym and "people_1/f1" in gym
    cfg.data.composition_manifest = None
    with pytest.raises(ValueError, match="composition_manifest"):
        GymMocapMixDataset(cfg)


def test_mix_aligned_component_share_and_collate(converted, monkeypatch):
    import unified_perception.data.gym_pseudo3d as gym_module

    root, _, _ = converted
    folder = root.parent / "composition_aligned"
    folder.mkdir()
    cfg = cache_config(root, cameras=3)
    cfg.data.provider = "gym_mocap_mix"
    cfg.data.composition_manifest = str(_manifest(folder))
    cfg.data.composition_original_fraction = 0.25
    cfg.data.gym_sample_fraction = 0.25
    cfg.data.gym_mix_aligned_dirs = ["unused"]
    cfg.data.gym_mix_aligned_fraction = 0.25
    cfg.model.position_range = [-6000., -12000., -200., 6000., 12000., 9000.]
    cfg.model.num_joints = 12

    class FakeBase:
        """Single-person provider sample as MultiViewPoseDataset returns it."""
        def __len__(self):
            return 3

        def __getitem__(self, index):
            valid = torch.zeros((8, cfg.data.max_people), dtype=torch.bool)
            valid[:, 0] = True
            return {"imgs": torch.zeros((8, 3, 3, 32, 32), dtype=torch.uint8),
                    "world2img": torch.eye(4).repeat(8, 3, 1, 1), "intrinsics": torch.eye(4).repeat(8, 3, 1, 1),
                    "gt_keypoints": torch.zeros((8, cfg.data.max_people, 12, 3)), "gt_valid": valid,
                    "seq": f"aligned/{index}", "start": 0}

    def fake_init(self, cfg, split="train", count=None, limit=None):
        self.base = FakeBase()
        self.indices = [i % 3 for i in range(count or 3)][:limit]

    monkeypatch.setattr(gym_module.AlignedSingleDataset, "__init__", fake_init)
    mix = GymMocapMixDataset(cfg)
    total = len(mix.mocap) / 0.5
    assert len(mix.gym) == max(1, round(total * .25)) and len(mix.aligned) == max(1, round(total * .25))
    assert len(mix) == len(mix.mocap) + len(mix.gym) + len(mix.aligned)
    aligned = mix[len(mix) - 1]
    assert aligned["instance_ids"][:, 0].eq(0).all() and aligned["instance_ids"][:, 1:].eq(-1).all()
    assert aligned["visibility"].shape == (8, 3, cfg.data.max_people)
    assert aligned["visibility"][:, :, 0].eq(1).all() and aligned["visibility"][:, :, 1:].eq(0).all()
    batch = collate_batch([mix[0], mix[len(mix.mocap)], aligned])  # composition + gym + aligned
    assert batch["instance_ids"].shape == (3, 8, cfg.data.max_people)
    cfg.data.gym_mix_aligned_fraction = 0.75
    with pytest.raises(ValueError, match="below one"):
        GymMocapMixDataset(cfg)


def test_recommended_position_range_covers_targets_with_margin():
    from unified_perception.tools.import_gym_pseudo3d import recommended_position_range

    assert recommended_position_range([-5405., -11288., -116.], [5444., 12047., 1729.]) == \
        [-6000., -12500., -200., 6000., 12500., 3000.]
    assert recommended_position_range([-100., -100., 0.], [100., 100., 100.]) == \
        [-6000., -12000., -200., 6000., 12000., 3000.]
    with pytest.raises(ValueError, match="floor"):
        recommended_position_range([0., 0., -500.], [1., 1., 1.])


def test_checkpoint_helpers_atomic_and_compile_prefix(tmp_path):
    from unified_perception.engine.train import _save_checkpoint, _strip_compile_prefix, _unwrap

    path = str(tmp_path / "last.pt")
    _save_checkpoint(path, {"step": 1})
    _save_checkpoint(path, {"step": 2})
    assert torch.load(path)["step"] == 2 and not (tmp_path / "last.pt.tmp").exists()
    assert _strip_compile_prefix({"_orig_mod.a.w": 1, "b": 2}) == {"a.w": 1, "b": 2}
    inner = torch.nn.Linear(2, 2)
    wrapped = torch.nn.Module()
    wrapped._orig_mod = inner
    assert _unwrap(wrapped) is inner and _unwrap(inner) is inner


def test_real_cache_projection_and_mask_integrity():
    root = REPO / "data_cache" / "gym_pseudo3d_20250505"
    if not (root / "index.json").exists():
        pytest.skip("Run the real-session converter to enable this integration check")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if index.get("format_version", 1) < 2:
        pytest.skip("Regenerate the real cache with observation masks")
    h, w = index["image_hw"]
    for split in ("train", "val"):
        folder = root / split / "gym_20250505"
        with np.load(folder / "poses.npz") as p, np.load(folder / "calib.npz") as c:
            cam = np.einsum("cij,fpj->fcpi", c["R"], p["root"]) + c["T"][None, :, None]
            proj = np.einsum("cij,fcpj->fcpi", c["K"], cam)
            uv = proj[..., :2] / np.maximum(proj[..., 2:3], 1e-8)
            inside = ((cam[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < w)
                      & (uv[..., 1] >= h // 8) & (uv[..., 1] < h - h // 8))
            # Identities need not appear in all four cameras; union is the supervised count domain.
            assert inside.any(1)[p["root_valid"]].mean() > .90
            # Roots should land inside (or near: partially visible people) the views that observed them.
            observed = p["observed"].transpose(0, 2, 1) & p["root_valid"][:, None]
            margin = w // 8
            near = ((cam[..., 2] > 0) & (uv[..., 0] >= -margin) & (uv[..., 0] < w + margin)
                    & (uv[..., 1] >= -margin) & (uv[..., 1] < h + margin))
            assert near[observed].mean() > .90
            np.testing.assert_array_equal(p["observed"].any(-1), p["valid"])
            assert not p["keypoint_valid"][~p["labeled"]].any()
            assert (p["keypoint_valid"] <= p["valid"][..., None]).all()
            bounds = np.array(WIDE_RANGE)
            assert (p["root"][p["root_valid"]] >= bounds[:3]).all()
            assert (p["root"][p["root_valid"]] <= bounds[3:]).all()
    cfg = load_config(str(REPO / "configs" / "gym_pseudo3d_eval.yaml"))
    cfg.data.gym_pseudo3d_root = str(root)
    train, val = GymPseudo3DProvider(cfg, "train"), GymPseudo3DProvider(cfg, "val")
    assert len(train.samples) == 185 * 4 and len(val.samples) == 7 * 4
    assert max(train.max_window_people, val.max_window_people) == cfg.data.max_people
