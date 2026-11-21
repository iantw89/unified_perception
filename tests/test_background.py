import json
from types import SimpleNamespace

import numpy as np
import pytest

from unified_perception.data.background import (
    EXTRACTION_METHOD, BackgroundMean, _foreground_candidates, foreground_mask, person_region,
)
from unified_perception.data.composition import Clip, load_clip
from unified_perception.tools import compose as tool


def geometry(offset=0):
    joints = np.zeros((12, 3), dtype=np.float32)
    joints[:, 0] = offset
    joints[:, 2] = 1000
    projection = np.eye(4, dtype=np.float32)
    projection[:3, :3] = [[40, 0, 16], [0, 40, 16], [0, 0, 1]]
    return joints, projection


def test_average_uses_only_label_certified_clean_pixels():
    accumulator = BackgroundMean((4, 4, 3))
    excluded = np.zeros((4, 4), dtype=bool)
    excluded[1, 1] = True
    first = np.full((4, 4, 3), 40, dtype=np.uint8)
    first[1, 1] = 200
    accumulator.update(first, excluded)
    second = np.full((4, 4, 3), 60, dtype=np.uint8)
    accumulator.update(second, np.zeros_like(excluded))
    mean, std, count = accumulator.arrays()
    np.testing.assert_array_equal(mean[0, 0], [50, 50, 50])
    np.testing.assert_array_equal(mean[1, 1], [60, 60, 60])
    np.testing.assert_array_equal(std[0, 0], [10, 10, 10])
    assert count[0, 0] == 2 and count[1, 1] == 1


def test_persistent_person_pixels_are_unknown_not_background():
    accumulator = BackgroundMean((4, 4, 3))
    excluded = np.zeros((4, 4), dtype=bool)
    excluded[1, 1] = True
    for _ in range(4):
        accumulator.update(np.full((4, 4, 3), 200, dtype=np.uint8), excluded)
    mean, std, count = accumulator.arrays()
    assert count[1, 1] == 0
    assert count[0, 0] == 4


def test_foreground_uses_label_roi_and_ignores_distant_motion():
    joints, projection = geometry()
    region, pixels = person_region(joints, projection, (32, 32), 100)
    rgb = np.full((32, 32, 3), 40, dtype=np.uint8)
    rgb[region] = 180
    rgb[:3, :3] = 255
    mask = foreground_mask(rgb, joints, projection, np.full_like(rgb, 40, dtype=np.float32),
                           np.zeros_like(rgb, dtype=np.float32),
                           np.full((32, 32), 3), margin_mm=100)
    assert not mask[:3, :3].any()
    assert mask[pixels[:, 1], pixels[:, 0]].all()


def test_insufficient_background_is_an_explicit_rejection():
    joints, projection = geometry()
    with pytest.raises(ValueError, match="Insufficient"):
        foreground_mask(np.full((32, 32, 3), 180, dtype=np.uint8), joints, projection,
                        np.zeros((32, 32, 3)), np.zeros((32, 32, 3)),
                        np.zeros((32, 32), dtype=int), margin_mm=100)


def test_noise_variance_does_not_make_false_foreground():
    joints, projection = geometry()
    with pytest.raises(ValueError, match="incomplete"):
        foreground_mask(np.full((32, 32, 3), 60, dtype=np.uint8), joints, projection,
                        np.full((32, 32, 3), 40), np.full((32, 32, 3), 10),
                        np.full((32, 32), 3), margin_mm=100)


def test_two_pass_preparation_and_template_provenance(tmp_path, monkeypatch):
    clips = {}
    for identity, offset in (("left", -250), ("right", 250)):
        joints, projection = geometry(offset)
        region, _ = person_region(joints, projection, (32, 32), 100)
        rgb = np.full((32, 32, 3), 40, dtype=np.uint8)
        rgb[region] = 180
        metadata = dict(id=identity, split="train", split_group=identity,
                        capture_group="rig", subject="actor", sequence=identity,
                        source_sequence="h36m:" + identity, camera_ids=["0"],
                        camera_frame_ids={"0": [0, 1, 2]}, frame_ids=[0, 1, 2],
                        fps=10., extraction_hash="pending-background", geometry_verified=True,
                        distortion="undistorted", approved=False, path=identity + ".npz")
        clips[identity] = Clip(metadata, np.tile(rgb.transpose(2, 0, 1), (3, 1, 1, 1, 1)),
                               np.tile(region, (3, 1, 1, 1)), np.tile(joints, (3, 1, 1)),
                               np.tile(projection, (3, 1, 1, 1)), np.tile(projection, (3, 1, 1, 1)))
    sequences = {i: SimpleNamespace(source="h36m", source_id=i) for i in clips}
    monkeypatch.setattr(tool, "HumanPriorAlignedProvider", lambda *args: SimpleNamespace(_seqs=sequences))
    monkeypatch.setattr(tool, "_source_clip", lambda provider, source, margin: clips[source["id"]])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(split_protocol="recording-isolated fixture",
                                   clips=[clip.metadata for clip in clips.values()])))
    root = tmp_path / "cache"
    tool.prepare(SimpleNamespace(config=None, spec=str(spec), output=str(root),
                                 background_margin_mm=100., min_background_count=3,
                                 pixel_threshold=20., noise_scale=3.))
    catalog = json.loads((root / "catalog.json").read_text())
    assert catalog["extraction_method"] == EXTRACTION_METHOD
    assert len(catalog["backgrounds"]) == 1
    assert len(catalog["clips"]) == 2
    assert not list(root.glob("*.raw.npz"))
    for entry in catalog["clips"]:
        clip = load_clip(root, entry)
        assert clip.masks.any()
        assert entry["background_sha256"] == catalog["backgrounds"][0]["sha256"]


def limb_fixture():
    joints = np.array([
        [-100, -100, 1000], [100, -100, 1000],
        [-100, -350, 1000], [100, -350, 1000],
        [-200, -200, 1000], [200, -200, 1000],
        [-250, -50, 1000], [250, -50, 1000],
        [-100, 100, 1000], [100, 100, 1000],
        [-100, 350, 1000], [100, 350, 1000],
    ], dtype=np.float32)
    projection = np.array([[100, 0, 50, 0], [0, 100, 50, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    mean = np.full((100, 100, 3), [100, 40, 30], dtype=np.float32)
    rgb = mean.astype(np.uint8)
    return joints, projection, mean, rgb


def test_brightness_only_shadow_removed_but_black_shoes_protected():
    joints, projection, mean, rgb = limb_fixture()
    # Red floor darkened without changing its color, outside the body core.
    rgb[86:91, 70:84] = [60, 24, 18]
    # Shoe neighbourhood: equally shadow-like values must not erase the shoe.
    rgb[83:89, 37:44] = [40, 16, 12]
    rgb[83:89, 57:64] = [8, 8, 8]
    region = np.ones((100, 100), dtype=bool)
    mask = _foreground_candidates(rgb, joints, projection, mean, np.ones_like(mean), region, 20., 3.)
    assert not mask[86:91, 70:84].any()
    assert mask[85, 40]
    assert mask[85, 60]


def test_low_contrast_calves_recovered_only_with_connected_image_evidence():
    joints, projection, mean, rgb = limb_fixture()
    # Strong knee and shoe are separated by a calf whose channels differ by
    # only 12 levels; the old uniform 20-level threshold cuts this out.
    rgb[58:64, 38:43] = [160, 80, 55]
    rgb[64:82, 38:43] = [100, 52, 32]
    rgb[82:87, 38:43] = [8, 8, 8]
    mask = _foreground_candidates(rgb, joints, projection, mean, np.ones_like(mean),
                                  np.ones((100, 100), dtype=bool), 20., 3.)
    assert mask[64:82, 39:42].all()
    # Geometry is not a license to paint a second leg where RGB is unchanged.
    assert not mask[65:80, 58:63].any()
    # Nor may disconnected weak changes elsewhere become foreground.
    rgb[68:73, 58:63] = [100, 52, 32]
    mask = _foreground_candidates(rgb, joints, projection, mean, np.ones_like(mean),
                                  np.ones((100, 100), dtype=bool), 20., 3.)
    assert not mask[68:73, 58:63].any()


def test_shadow_suppression_does_not_remove_changed_color_foreground():
    joints, projection, mean, rgb = limb_fixture()
    rgb[30:40, 75:85] = [45, 45, 45]
    mask = _foreground_candidates(rgb, joints, projection, mean, np.ones_like(mean),
                                  np.ones((100, 100), dtype=bool), 20., 3.)
    assert mask[30:40, 75:85].all()


def test_shadow_like_clothing_away_from_feet_is_not_removed():
    joints, projection, mean, rgb = limb_fixture()
    rgb[28:37, 68:77] = [60, 24, 18]
    mask = _foreground_candidates(rgb, joints, projection, mean, np.ones_like(mean),
                                  np.ones((100, 100), dtype=bool), 20., 3.)
    assert mask[28:37, 68:77].all()
