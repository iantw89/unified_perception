from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent

from unified_perception.config import Config
from unified_perception.data import MultiViewPoseDataset
from unified_perception.tools import training_report as report


def test_default_cli_and_overrides():
    defaults = report.build_parser().parse_args([])
    assert defaults.run_dir == Path("runs") / "gym_pseudo3d_full_v4"
    assert defaults.checkpoint == Path("last.pt")
    assert defaults.output is None and defaults.config is None
    assert defaults.device == "auto" and defaults.examples == 4
    assert defaults.score_threshold == 0.3
    assert not defaults.gym_only and not defaults.no_animations
    args = report.build_parser().parse_args([
        "--run-dir", "another-run", "--checkpoint", "best_val_gym.pt",
        "--device", "cpu", "--examples", "2", "--gym-only", "--no-animations",
    ])
    assert args.checkpoint.name == "best_val_gym.pt" and args.gym_only
    assert args.examples == 2 and args.device == "cpu" and args.no_animations


@pytest.mark.parametrize("args", [["--examples", "0"], ["--score-threshold", "1.1"],
                                  ["--score-threshold", "nan"]])
def test_invalid_arguments_fail_before_inference(args):
    with pytest.raises(SystemExit):
        report.main(args)


@pytest.mark.parametrize("length,count,expected", [(7, 4, [0, 2, 4, 6]),
                                                   (2, 4, [0, 1]), (1, 4, [0])])
def test_even_indices_are_unique_and_stable(length, count, expected):
    assert report.even_indices(length, count) == expected


def test_snapshot_loads_cpu_weights_and_metadata(tmp_path):
    path = tmp_path / "last.pt"
    torch.save({"model": {"weight": torch.ones(2)}, "optimizer_steps": 99}, path)
    state, modified = report.load_snapshot(path)
    assert state["model"]["weight"].device.type == "cpu"
    assert state["optimizer_steps"] == 99
    assert modified.endswith("+00:00")
    with pytest.raises(FileNotFoundError):
        report.load_snapshot(tmp_path / "missing.pt")


def test_failed_refresh_preserves_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "visualizations"
    output.mkdir()
    previous = output / "index.html"
    previous.write_text("old report")

    def fail(args, staged):
        (staged / "index.html").write_text("incomplete report")
        raise RuntimeError("inference failed")

    monkeypatch.setattr(report, "_generate_report", fail)
    args = report.build_parser().parse_args(["--output", str(output)])
    with pytest.raises(RuntimeError, match="inference failed"):
        report.generate_report(args)
    assert previous.read_text() == "old report"
    assert list(output.iterdir()) == [previous]


def test_successful_refresh_publishes_staged_files(tmp_path, monkeypatch):
    def generate(args, staged):
        (staged / "index.html").write_text("new report")
        (staged / "summary.json").write_text("{}")

    monkeypatch.setattr(report, "_generate_report", generate)
    args = report.build_parser().parse_args(["--output", str(tmp_path)])
    assert report.generate_report(args).read_text() == "new report"
    assert (tmp_path / "summary.json").read_text() == "{}"


def test_restart_scalars_keep_latest_wall_time(tmp_path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / f"events.out.tfevents.{name}").touch()
    streams = {
        "a": {"train/loss": [ScalarEvent(10, 1, 9), ScalarEvent(30, 2, 3)]},
        "b": {"train/loss": [ScalarEvent(20, 1, 4), ScalarEvent(25, 2, 8)],
              "val_gym/f1": [ScalarEvent(22, 1, 0.8)]},
    }

    class Events:
        def __init__(self, path, size_guidance):
            self.data = streams[Path(path).suffix[1:]]

        def Reload(self):
            return self

        def Tags(self):
            return {"scalars": list(self.data)}

        def Scalars(self, tag):
            return self.data[tag]

    monkeypatch.setattr(report, "EventAccumulator", Events)
    tags = report.read_scalars(tmp_path)
    assert tags["train/loss"][1].value == 4
    assert tags["train/loss"][2].value == 3
    assert tags["val_gym/f1"][1].value == 0.8


def test_training_plots_handle_missing_and_validation_scalars(tmp_path):
    with pytest.warns(UserWarning, match="No saved training loss"):
        logs, assets = report.plot_training({}, tmp_path, tmp_path, 3500)
    assert logs["latest_logged_step"] is None
    assert not logs["epoch_validation_logged"]
    assert assets == ["learning_curves.png"]
    tags = {"train/loss": {10: ScalarEvent(1, 10, 1)},
            "val_gym/mpjpe_mm": {9: ScalarEvent(1, 9, 200)},
            "val_gym/f1": {9: ScalarEvent(1, 9, 0.8)}}
    logs, assets = report.plot_training(tags, tmp_path, tmp_path, 10)
    assert logs["latest_logged_step"] == 10
    assert logs["latest_epoch_validation"]["val_gym"]["global_step"] == 9
    assert "validation_history.png" in assets
    assert (tmp_path / "validation_history.png").is_file()


def test_composition_examples_stratify_by_source_and_extreme_counts(monkeypatch):
    entries = [{"dataset": "h36m"}, {"dataset": "mpi"}]
    recipes = [{"sources": [source] * count}
               for source in (0, 1) for count in (1, 2, 4) for _ in range(3)]
    dataset = MultiViewPoseDataset.__new__(MultiViewPoseDataset)
    monkeypatch.setattr(dataset, "composed", SimpleNamespace(entries=entries, recipes=recipes), raising=False)
    assert report.compose_indices(dataset) == [1, 7, 10, 16]


def sample_at_size(size):
    points = torch.ones(1, 1, 12, 3)
    points[..., 0] = 2 * (size - 50)
    points[..., 1] = 2 * (size - 50)
    points[..., 2] = 2
    return {
        "imgs": torch.zeros(1, 1, 3, size, size, dtype=torch.uint8),
        "world2img": torch.eye(4)[None, None], "frame_ids": torch.tensor([240]),
        "gt_valid": torch.ones(1, 1, dtype=torch.bool), "gt_keypoints": points,
        "gt_keypoint_valid": torch.zeros(1, 1, 12, dtype=torch.bool),
        "gt_keypoint_frame_valid": torch.zeros(1, dtype=torch.bool),
        "gt_root": points[:, :, 0], "gt_root_valid": torch.ones(1, 1, dtype=torch.bool),
        "instance_ids": torch.zeros(1, 1, dtype=torch.long),
    }


@pytest.mark.parametrize("size", [320, 640])
def test_sparse_rendering_resolution_roots_and_threshold(size):
    cfg = Config()
    sample = sample_at_size(size)
    scores = np.array([[0.4, 0.6]])
    poses = np.tile(sample["gt_keypoints"].numpy(), (1, 2, 1, 1))
    frames, details, metrics = report.render_clip(sample, scores, poses, cfg, [0], "Test", 0.5)
    assert frames[0].shape == (size + 50, size * 2, 3)
    assert details[0]["predicted"] == details[0]["matched"] == 1
    assert metrics["f1"] == 1 and np.isnan(metrics["mpjpe_mm"])
    frames, details, _ = report.render_clip(sample, scores, poses, cfg, [0], "Test", 0.7)
    assert details[0]["missed"] == 1
    # With no predictions or valid GT joints, this can only be the GT root marker.
    assert frames[0][50 + size - 60:50 + size - 30, size - 60:size - 30].max() > 0


def test_undefined_metrics_display_na(tmp_path):
    cfg = Config()
    metrics = {"mpjpe_mm": None, "root_error_mm": None, "precision": 0, "recall": 0,
               "f1": 0, "pck@50mm": None, "pck@100mm": None, "pck@150mm": None}
    report.plot_validation(metrics, cfg, 7, tmp_path)
    assert (tmp_path / "gym_validation.png").is_file()
    assert report.metric_text(None) == "n/a"
    assert report.finite_metrics({"mpjpe_mm": float("nan")}) == {"mpjpe_mm": None}


def test_html_uses_dynamic_progress_history_and_escaped_strings(tmp_path):
    summary = {
        "run": str(tmp_path / "test_run"), "checkpoint": str(tmp_path / "best_val_gym.pt"),
        "checkpoint_metadata": {"epoch": 4, "epoch_complete": True, "optimizer_steps": 12345,
                                "mtime_utc": "now"},
        "target_optimizer_steps": 100000, "generated_utc": "now", "device": "cpu",
        "training_logs": {"latest_logged_step": 12400, "latest_epoch_validation": {
            "val_gym": {"global_step": 12000, "metrics": {"mpjpe_mm": 190, "f1": 0.8}}}},
        "data_shape": {"height": 640, "width": 640, "cameras": 4, "frames": 8},
        "gym_validation_windows": 7, "gym_validation_window_frames": 56,
        "gym_validation": {"mpjpe_mm": None, "f1": 0},
        "score_threshold": 0.5, "match_threshold_mm": 600,
        "refresh_command": "python --output <report>",
        "examples": [{"group": "gym", "sequence": "<script>", "file": "still.png",
                      "animation": None, "metrics": {"f1": 0, "mpjpe_mm": None,
                                                     "root_error_mm": None}}],
    }
    report.write_report(tmp_path, summary, ["learning_curves.png"])
    page = (tmp_path / "index.html").read_text()
    assert "12,345 / 100,000" in page and "epoch 4 complete" in page
    assert "12400" in page and "12000" in page
    assert "No epoch-end" not in page and "paused" not in page
    assert "&lt;script&gt;" in page and "&lt;report&gt;" in page
    assert "src='still.png'" in page and "n/a" in page
