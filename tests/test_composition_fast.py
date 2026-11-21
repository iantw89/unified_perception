import hashlib

import numpy as np
import pytest

from unified_perception.data.composition import (
    compose, load_clip, save_clip, verify_clip_assets,
)
from test_composition import BOUNDS, make_clip


def test_lossless_mmap_roundtrip_and_asset_safety(tmp_path):
    clip = make_clip()
    clip.metadata["path"] = "a.npz"
    save_clip(tmp_path, clip, storage="npy")
    loaded = load_clip(tmp_path, clip.metadata)
    assert isinstance(loaded.imgs, np.memmap)
    assert not loaded.imgs.flags.writeable
    for name in ("imgs", "masks", "joints", "world2img", "intrinsics"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(clip, name))
    verify_clip_assets(tmp_path, clip.metadata)
    asset = tmp_path / clip.metadata["imgs_path"]
    with asset.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Stale"):
        verify_clip_assets(tmp_path, clip.metadata)


def test_geometry_only_preserves_visibility_and_order():
    clips = [make_clip("near"), make_clip("far", depth=4000)]
    images, visible, order = compose(clips, BOUNDS)
    skipped, fast_visible, fast_order = compose(clips, BOUNDS, geometry_only=True)
    assert skipped is None
    np.testing.assert_array_equal(visible, fast_visible)
    np.testing.assert_array_equal(order, fast_order)
    assert hashlib.sha256(order.tobytes()).digest() == hashlib.sha256(fast_order.tobytes()).digest()


def test_mmap_composite_is_pixel_exact(tmp_path):
    clips = [make_clip("near", color=31), make_clip("far", depth=4000, color=199)]
    clips[0].masks[..., 4:12, 8:12] = False
    expected = compose(clips, BOUNDS)
    mapped = []
    for clip in clips:
        clip.metadata["path"] = clip.metadata["id"] + ".npz"
        save_clip(tmp_path, clip, storage="npy")
        mapped.append(load_clip(tmp_path, clip.metadata))
    actual = compose(mapped, BOUNDS)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_fast_background_statistics_are_exact():
    from unified_perception.data.background import BackgroundMean
    rng = np.random.default_rng(4)
    accumulator = BackgroundMean((12, 13, 3))
    total = np.zeros((12, 13, 3), dtype=np.float64)
    squares = total.copy()
    count = np.zeros((12, 13), dtype=np.uint32)
    for _ in range(12):
        image = rng.integers(0, 256, total.shape, dtype=np.uint8)
        excluded = rng.random(count.shape) < .3
        accumulator.update(image, excluded)
        values = image.astype(np.float64)
        total += values * (~excluded)[..., None]
        squares += values * values * (~excluded)[..., None]
        count += ~excluded
    np.testing.assert_array_equal(accumulator.sum, total)
    np.testing.assert_array_equal(accumulator.sum_squares, squares)
    np.testing.assert_array_equal(accumulator.count, count)


def test_build_only_freezes_manifest_without_training(tmp_path, monkeypatch):
    import json
    from unified_perception.config import Config, save_config
    from unified_perception.tools import composition_experiment as experiment
    config = tmp_path / "config.yaml"
    save_config(Config(), str(config))
    output = tmp_path / "build"
    monkeypatch.setattr("sys.argv", ["composition_experiment", "--config", str(config),
                                   "--output", str(output), "--build-only"])
    monkeypatch.setenv("FFMPEG_SHARED_BIN", str(tmp_path))
    monkeypatch.setattr(experiment, "prepare_corpus", lambda *a, **k: {"clips": []})
    frozen = []
    def freeze(catalog, root, *args, **kwargs):
        path = root / "experiment.json"
        experiment.write_json(path, {"clips": []})
        frozen.append(path)
        return path
    def forbidden(*args, **kwargs):
        pytest.fail("Build-only must not train or evaluate")
    monkeypatch.setattr(experiment, "freeze_recipes", freeze)
    monkeypatch.setattr(experiment, "run_command", forbidden)
    monkeypatch.setattr(experiment, "evaluate_arm", forbidden)
    experiment.main()
    assert frozen == [output.resolve() / "cache" / "experiment.json"]
    status = json.loads((output / "status.json").read_text())
    assert status["stage"] == "built"
    assert status["training_started"] is False


def test_writable_masks_can_change_between_compositions():
    near = make_clip("near", color=31)
    far = make_clip("far", depth=4000, color=199)
    compose([near, far], BOUNDS, blend_intersections=False)
    far.masks[..., 0, 0] = True
    image, _, _ = compose([near, far], BOUNDS, blend_intersections=False)
    assert (image[..., 0, 0] == 199).all()


def test_materialized_recipe_matches_live_loader_and_resumes(tmp_path):
    import json
    from unified_perception.config import Config
    from unified_perception.data.composition import VERSION, CompositionDataset
    from unified_perception.tools.composition_experiment import materialize_recipes
    clips = [make_clip("near", color=31), make_clip("far", depth=4000, color=199)]
    clips[0].masks[..., 4:12, 8:12] = False
    for clip in clips:
        clip.metadata["path"] = clip.metadata["id"] + ".npz"
        save_clip(tmp_path, clip, storage="npy")
    order = compose(clips, BOUNDS)[2]
    recipe = dict(id="train-0", split="train", sources=[c.metadata["id"] for c in clips],
                  order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), approved=True)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(dict(version=VERSION, quality_approved=True, clips=[c.metadata for c in clips],
                                   recipes=[recipe], parameters=dict(position_range=BOUNDS))))
    cfg = Config()
    cfg.data.composition_manifest = str(path)
    cfg.data.seq_len, cfg.data.num_cameras = 3, 2
    cfg.data.image_height = cfg.data.image_width = 16
    cfg.model.position_range = BOUNDS
    expected = CompositionDataset(cfg, "train")[0]
    materialize_recipes(path, lambda *a, **k: None)
    actual = CompositionDataset(cfg, "train")[0]
    for key in ("imgs", "visibility", "world2img", "intrinsics", "gt_keypoints", "gt_valid", "instance_ids"):
        np.testing.assert_array_equal(expected[key].numpy(), actual[key].numpy())
    metadata = json.loads(path.read_text())["recipes"][0]["pixel_cache"]
    materialize_recipes(path, lambda *a, **k: None)
    assert json.loads(path.read_text())["recipes"][0]["pixel_cache"] == metadata
    changed = json.loads(path.read_text())
    changed["recipes"][0]["sources"].reverse()
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Cached pixels differ"):
        CompositionDataset(cfg, "train")
