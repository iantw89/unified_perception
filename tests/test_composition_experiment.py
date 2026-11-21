import json
from types import SimpleNamespace

import numpy as np
import pytest

from unified_perception.config import Config
from unified_perception.data.composition import generate_recipes
from unified_perception.tools import composition_experiment as experiment
from test_composition import BOUNDS, make_clip, write_cache


@pytest.mark.parametrize("dataset,source,split", [
    ("h36m", "subj_1_act_10_sub_1", "train"),
    ("h36m", "subj_9_act_10_sub_1", "val"),
    ("h36m", "subj_11_act_14_sub_1", "test"),
    ("mpi", "S1_Seq1", "train"),
    ("mpi", "S7_Seq2", "val"),
    ("mpi", "S8_Seq1", "test"),
])
def test_subject_splits(dataset, source, split):
    assert experiment.source_split(dataset, source) == split
    assert experiment.source_split(dataset, source + "__seg02_100_200") == split


@pytest.mark.parametrize("dataset,source", [("moyo", "anything"), ("h36m", "subj_99"), ("mpi", "S9")])
def test_unexpected_subjects_are_not_silently_training(dataset, source):
    with pytest.raises(ValueError):
        experiment.source_split(dataset, source)


def test_preloaded_recipe_cache_matches_disk_and_preserves_all_entries(tmp_path):
    clips = {clip.metadata["id"]: clip for clip in [
        make_clip("a"), make_clip("b", depth=4000), make_clip("c", depth=6500),
    ]}
    entries = [write_cache(tmp_path, clip) for clip in clips.values()]
    args = entries, tmp_path, "train", 3, [2], 5, BOUNDS
    assert generate_recipes(*args, max_attempts=20) == generate_recipes(
        *args, max_attempts=20, clip_cache=clips,
    )
    assert set(clips) == {"a", "b", "c"}


def test_preparation_records_rejections_without_fallback_masks(tmp_path, monkeypatch):
    seq = SimpleNamespace(source="h36m", source_id="subj_1_act_1_sub_1", num_frames=8)
    provider = SimpleNamespace(
        _seqs={"seq": seq}, num_cameras=2, seq_len=3, out_hw=(16, 16), decoders={},
        cfg=Config(),
        _cam_defs_for_seq=lambda seq: {"1": {}, "2": {}},
    )

    def source_clip(provider, source, margin):
        if source["start"] == 0:
            raise ValueError("missing joints")
        clip = make_clip(source["id"])
        clip.metadata.update(source, path=source["id"] + ".npz",
                             camera_frame_ids={"1": [0, 1, 2], "2": [0, 1, 2]})
        return clip

    monkeypatch.setattr(experiment, "_source_clip", source_clip)
    monkeypatch.setattr(experiment, "foreground_mask",
                        lambda *args: np.ones((16, 16), dtype=bool))
    report = experiment.prepare_sequence(provider, "seq", tmp_path, 2)
    assert report["requested"] == 2
    assert report["accepted"] == 1
    assert report["failures"][0]["errors"] == ["missing joints"]
    assert len(list(tmp_path.glob("h36m-*.npz"))) == 1
    assert provider.decoders == {}


@pytest.mark.parametrize("frames,cameras", [(20, {"2": {}, "3": {}, "4": {}}),
                                          (2, {"1": {}, "2": {}, "3": {}})])
def test_unusable_sequence_is_explicitly_reported_and_excluded(tmp_path, frames, cameras):
    provider = SimpleNamespace(
        _seqs={"seq": SimpleNamespace(source="h36m", source_id="subj_11_act_2_sub_2",
                                     num_frames=frames)},
        num_cameras=3, seq_len=8, decoders={"old": object()},
        _cam_defs_for_seq=lambda seq: cameras,
    )
    result = experiment.prepare_sequence(provider, "seq", tmp_path, 24)
    assert result["accepted"] == result["decoded"] == 0
    assert result["clips"] == []
    assert "Insufficient views or frames" in result["failures"][0]["errors"][0]
    assert result["split"] == "test"
    assert provider.decoders == {}
    assert not list(tmp_path.iterdir())


def test_infeasible_counts_are_reported_not_substituted(tmp_path, monkeypatch):
    entries = []
    for split in ("train", "val", "test"):
        for dataset in ("h36m", "mpi"):
            for index in range(2):
                clip = make_clip(f"{split}-{dataset}-{index}", split=split, depth=1500 + 2500 * index)
                clip.metadata.update(capture_group=f"{split}-{dataset}", dataset=dataset)
                entries.append(write_cache(tmp_path, clip))
    cfg = Config()
    cfg.model.position_range = BOUNDS
    path = experiment.freeze_recipes(dict(clips=entries), tmp_path, cfg, 1, 8, lambda *a, **k: None)
    document = json.loads(path.read_text())
    assert {len(r["sources"]) for r in document["recipes"]} == {2}
    assert document["reports"]["train-h36m"]["people_8"]["achieved"] == 0
    assert document["reports"]["train-h36m"]["people_8"]["requested"] > 0
    assert document["provisional_quality"] is True


def test_acceptance_leaves_undefined_pose_metrics_undefined():
    results = {
        "baseline": {"test/original": {"f1": .8, "mpjpe_mm": 100},
                     "test/synthetic": {"f1": .1, "mpjpe_mm": None}},
        "augmented_10000": {"test/original": {"f1": .79, "mpjpe_mm": 104},
                            "test/synthetic": {"f1": .2, "mpjpe_mm": 120}},
        "augmented": {"test/original": {"f1": .7, "mpjpe_mm": 110},
                      "test/synthetic": {"f1": .3, "mpjpe_mm": 120}},
    }
    gates = experiment.acceptance(results)
    matched = gates["step_matched"]
    assert matched["candidate"] == "augmented_10000"
    assert matched["single_person_f1"] is True
    assert matched["single_person_mpjpe"] is True
    assert matched["synthetic_f1"] is True
    assert matched["synthetic_mpjpe"] is None
    longer = gates["longer_augmented"]
    assert longer["single_person_f1"] is False
    assert longer["single_person_mpjpe"] is False
    assert gates["real_multi_person_claim"] is False


def test_stages_snapshot_augmented_at_baseline_budget():
    stages = experiment.experiment_stages(10000)
    assert [(name, directory, steps) for name, directory, steps, _, _ in stages] == [
        ("baseline", "baseline", 10000),
        ("augmented_10000", "augmented", 10000),
        ("augmented", "augmented", 30000),
    ]
    assert len({metrics for _, directory, _, _, metrics in stages if directory == "augmented"}) == 2


def test_official_masks_bypass_background_gate_and_pool_only_compatible_segments(tmp_path, monkeypatch):
    provider = SimpleNamespace(
        _seqs={name: SimpleNamespace(source="mpi", source_id="S1_Seq1", num_frames=3)
               for name in ("first-segment", "second-segment")},
        num_cameras=2, seq_len=3, out_hw=(16, 16), decoders={}, cfg=Config(),
        _cam_defs_for_seq=lambda _: {"0": {}, "2": {}},
    )

    def source_clip(provider, source, margin):
        clip = make_clip(source["id"])
        clip.metadata.update(source, dataset="mpi", mask_method="mpi-official-v1",
                             extraction_hash="official-asset-hash", path=source["id"] + ".npz",
                             camera_frame_ids={"0": [1, 2, 3], "2": [1, 2, 3]})
        return clip

    monkeypatch.setattr(experiment, "_source_clip", source_clip)

    def forbidden(*args):
        pytest.fail("Official masks must not require background support")

    monkeypatch.setattr(experiment, "foreground_mask", forbidden)
    first = experiment.prepare_sequence(provider, "first-segment", tmp_path, 1, cross_segment=True)
    second = experiment.prepare_sequence(provider, "second-segment", tmp_path, 1, cross_segment=True)
    assert first["accepted"] == second["accepted"] == 1
    a, b = first["clips"][0], second["clips"][0]
    assert a["id"] != b["id"]
    assert a["capture_group"] == b["capture_group"]
    assert a["extraction_hash"] == b["extraction_hash"] == "official-asset-hash"
    with np.load(tmp_path / a["path"]) as arrays:
        np.testing.assert_array_equal(arrays["masks"], make_clip().masks)


def test_camera_sets_get_unique_cache_ids():
    assert experiment.preparation_tag("seq", ["0", "2", "4"]) != experiment.preparation_tag("seq", ["1", "5", "6"])
    assert experiment.preparation_tag("seq") != experiment.preparation_tag("seq", ["0", "2", "4"])


def test_prepare_only_cannot_start_recipes_training_or_evaluation(tmp_path, monkeypatch):
    from unified_perception.config import save_config

    config = tmp_path / "config.yaml"
    save_config(Config(), str(config))
    output = tmp_path / "preparation"
    monkeypatch.setattr("sys.argv", [
        "composition_experiment", "--config", str(config), "--output", str(output),
        "--prepare-only", "--mpi-masks", "prefer", "--mpi-windows-per-sequence", "96",
        "--mpi-camera-sets", "0,2,4,6", "1,5,7,8", "--mpi-cross-segment",
    ])
    monkeypatch.setenv("FFMPEG_SHARED_BIN", str(tmp_path))
    observed = {}

    def prepare(cfg, root, windows, status, **kwargs):
        observed.update(kwargs, masks=cfg.data.composition_mpi_masks)
        return {"clips": []}

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation-only must not generate recipes, train or evaluate")

    monkeypatch.setattr(experiment, "prepare_corpus", prepare)
    monkeypatch.setattr(experiment, "freeze_recipes", forbidden)
    monkeypatch.setattr(experiment, "run_command", forbidden)
    monkeypatch.setattr(experiment, "evaluate_arm", forbidden)
    experiment.main()
    assert observed == dict(mpi_windows=96, mpi_camera_sets=[["0", "2", "4", "6"], ["1", "5", "7", "8"]],
                            cross_segment=True, masks="prefer")
    status = json.loads((output / "status.json").read_text())
    assert status["stage"] == "prepared"
    assert status["training_started"] is False
