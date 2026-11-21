from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from unified_perception.config import Config
from unified_perception.data.mpi_masks import (
    apply_official_masks, capture_group, close_mask_readers, mask_paths, person_mask, recording_ids,
)
from unified_perception.tools.download_mpi_masks import install_archive
from test_composition import make_clip


def mpi_clip():
    clip = make_clip()
    clip.metadata.update(dataset="mpi", sequence="segment", camera_ids=["0", "2"],
                         camera_frame_ids={"0": [7, 9, 11], "2": [6, 8, 10]})
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = 100
    intrinsic[0, 2] = intrinsic[1, 2] = 8
    clip.intrinsics[:] = intrinsic
    clip.world2img[:] = intrinsic
    return clip


def test_official_red_channels_are_keep_masks_not_chair_foreground():
    clip = mpi_clip()
    fg = np.zeros((16, 16, 3), dtype=np.uint8)
    fg[3:13, 3:13, 0] = 255
    # Clothing channels are deliberately unrelated to the silhouette.
    fg[..., 1:] = 255
    chair = np.zeros_like(fg)
    chair[..., 0] = 255
    chair[3:6, 3:6, 0] = 0
    mask = person_mask(fg, chair, clip.joints[0], clip.world2img[0, 0])
    assert mask[8, 8]
    assert not mask[4, 4]
    assert not mask[0, 0]
    assert mask.sum() == 91
    with pytest.raises(ValueError, match="incomplete"):
        person_mask(np.zeros_like(fg), chair, clip.joints[0], clip.world2img[0, 0])


def test_recording_paths_strip_segment_suffix(tmp_path):
    assert recording_ids("S8_Seq1__seg02_5116_6466") == (8, 1)
    fg, chair = mask_paths(tmp_path, "S8_Seq1__seg02_5116_6466", "4")
    assert fg == tmp_path / "S8" / "Seq1" / "FGmasks" / "video_4.avi"
    assert chair.parent.name == "ChairMasks"
    with pytest.raises(ValueError):
        recording_ids("S9_Seq1")


@pytest.mark.parametrize("source", ["S8_Seq1", "S1_Seq2", "S4_Seq2"])
def test_only_missing_or_known_bad_assets_allow_logged_fallback(tmp_path, caplog, source):
    cfg = Config()
    cfg.data.human_prior_mpi_dataset_path = str(tmp_path)
    cfg.data.composition_mpi_masks = "prefer"
    provider = SimpleNamespace(cfg=cfg, _seqs={"segment": SimpleNamespace(source_id=source, seq_id="segment")})
    clip = mpi_clip()
    assert apply_official_masks(provider, clip) is False
    assert "using label-gated background extraction" in caplog.text
    assert clip.metadata["official_mask_fallback"]
    cfg.data.composition_mpi_masks = "required"
    with pytest.raises(ValueError):
        apply_official_masks(provider, clip)


def test_video_indices_geometry_and_reader_cleanup(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data.human_prior_mpi_dataset_path = str(tmp_path)
    cfg.data.composition_mpi_masks = "required"
    provider = SimpleNamespace(
        cfg=cfg, out_hw=(16, 16),
        _seqs={"segment": SimpleNamespace(source_id="S8_Seq1", seq_id="segment")},
        _cam_defs_for_seq=lambda _: {"0": {}, "2": {}},
    )
    for camera in ("0", "2"):
        for path in mask_paths(tmp_path, "S8_Seq1", camera):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test mask")
    seeks, readers = [], []

    class Reader:
        def __init__(self, path):
            self.path, self.closed = Path(path), False
            readers.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            return 16

        def set(self, prop, index):
            seeks.append((self.path.parent.name, self.path.name, index))
            return True

        def read(self):
            bgr = np.zeros((16, 16, 3), dtype=np.uint8)
            bgr[..., 2] = 255
            return True, bgr

        def release(self):
            self.closed = True

    monkeypatch.setattr(cv2, "VideoCapture", Reader)
    clip = mpi_clip()
    assert apply_official_masks(provider, clip)
    assert clip.metadata["mask_method"] == "mpi-official-v1"
    assert len(clip.metadata["mask_assets"]) == 4
    assert [i for kind, camera, i in seeks if kind == "FGmasks" and camera == "video_0.avi"] == [7, 9, 11]
    assert [i for kind, camera, i in seeks if kind == "ChairMasks" and camera == "video_2.avi"] == [6, 8, 10]
    close_mask_readers(provider)
    assert all(r.closed for r in readers)


def test_cross_segment_groups_require_exact_geometry_cameras_recording_and_fps():
    clip = mpi_clip()
    first = capture_group(clip, "S8_Seq1__seg00_0_624")
    assert capture_group(clip, "S8_Seq1__seg02_5116_6466") == first
    assert capture_group(clip, "S7_Seq1") != first
    clip.world2img[..., 0, 3] += .001
    assert capture_group(clip, "S8_Seq1") != first
    clip.world2img[..., 0, 3] -= .001
    clip.metadata["fps"] = 20.
    assert capture_group(clip, "S8_Seq1") != first
    clip.metadata["fps"] = 10.
    clip.metadata["camera_ids"].reverse()
    assert capture_group(clip, "S8_Seq1") != first


def test_prefer_does_not_hide_corrupt_official_videos(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data.human_prior_mpi_dataset_path = str(tmp_path)
    cfg.data.composition_mpi_masks = "prefer"
    provider = SimpleNamespace(cfg=cfg, _seqs={
        "segment": SimpleNamespace(source_id="S8_Seq1", seq_id="segment")},
        _cam_defs_for_seq=lambda _: {"0": {}, "2": {}})
    for camera in ("0", "2"):
        for path in mask_paths(tmp_path, "S8_Seq1", camera):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"corrupt")

    class Unreadable:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Unreadable())
    clip = mpi_clip()
    with pytest.raises(ValueError, match="Cannot open MPI mask video"):
        apply_official_masks(provider, clip)
    assert "official_mask_fallback" not in clip.metadata


def test_archive_install_is_idempotent_and_never_overwrites_existing_masks(tmp_path):
    archive = tmp_path / "masks.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("FGmasks/video_0.avi", b"official mask")
    output = tmp_path / "installed"
    assert install_archive(archive, output) == install_archive(archive, output)
    (output / "video_0.avi").write_bytes(b"different")
    with pytest.raises(FileExistsError):
        install_archive(archive, output)
    assert (output / "video_0.avi").read_bytes() == b"different"


def test_archive_rejects_non_video_members(tmp_path):
    archive = tmp_path / "masks.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("../../unrelated.txt", b"not a video")
    with pytest.raises(ValueError, match="Unexpected"):
        install_archive(archive, tmp_path / "installed")
    assert not (tmp_path / "unrelated.txt").exists()
