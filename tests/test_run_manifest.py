import json
import random

from unified_perception.config import Config
from unified_perception.data.streaming import ChunkPolicy, StreamComponent, StreamingBatchSampler
from unified_perception.engine import run_manifest


class _Loader:
    def __init__(self, sampler=None, n=3):
        self.batch_sampler, self.dataset, self.n = sampler, object(), n

    def __len__(self):
        return self.n


def _loaders():
    gym = StreamComponent("gym", [[("g", i) for i in range(40)]], lambda k: k, 0.5)
    mocap = StreamComponent("mocap", [[("m", i)] for i in range(30)], lambda k: k, 0.35)
    aligned = StreamComponent("aligned", [[("a", i) for i in range(10)] for _ in range(3)], lambda k: k, 0.15)
    policies = [ChunkPolicy([16, 0], [0.5, 0.5]), ChunkPolicy([1], [1.0]), ChunkPolicy([1, 4], [0.5, 0.5])]
    sampler = StreamingBatchSampler([gym, mocap, aligned], 4, 10, policies, seed=0)
    return {"train": _Loader(sampler), "val_gym": _Loader(n=7)}


def _cfg(tmp_path, **train):
    cfg = Config()
    cfg.train.output_dir = str(tmp_path / "run")
    cfg.data.provider = "human_prior_aligned"
    for key, value in train.items():
        setattr(cfg.train, key, value)
    return cfg


def test_record_contents(tmp_path):
    source = tmp_path / "exp.yaml"
    source.write_text("# rationale kept\ntrain:\n  lr: 0.001\n")
    cfg = _cfg(tmp_path)
    path = run_manifest.write_launch_record(cfg, _loaders(), config_path=str(source), argv=["train"])
    record = json.loads(path.read_text())
    assert record["launch"]["index"] == 1
    assert set(record["sections"]) >= {"streaming", "mixture", "reid", "gt_noise", "tracking"}
    assert record["sections"]["mixture"]["data.gym_sample_fraction"] == cfg.data.gym_sample_fraction
    train = record["sampler"]["train"]
    assert train["lane_owner"] == ["gym", "gym", "mocap", None]
    assert {c["name"]: c["windows"] for c in train["components"]} == {"gym": 40, "mocap": 30, "aligned": 30}
    assert record["sampler"]["eval"]["val_gym"]["batches"] == 7
    copy = path.parent / record["launch"]["config_source_copy"]
    assert copy.read_text().startswith("# rationale kept")
    assert record["config"]["train"]["output_dir"] == cfg.train.output_dir


def test_launches_are_append_only_with_changes(tmp_path):
    cfg = _cfg(tmp_path)
    first = run_manifest.write_launch_record(cfg, _loaders())
    original = first.read_text()
    cfg.train.lr = 1e-3
    cfg.train.resume = str(tmp_path / "run" / "last.pt")
    second = run_manifest.write_launch_record(cfg, _loaders(), resumed_optimizer_steps=500)
    assert first != second and first.read_text() == original
    index = json.loads((tmp_path / "run" / "experiment.json").read_text())
    assert [l["record"] for l in index["launches"]] == [first.name, second.name]
    assert index["launches"][0]["config_changes_vs_previous"] is None
    changes = index["launches"][1]["config_changes_vs_previous"]
    assert set(changes) == {"train.lr"}  # resume path is expected to change and is not reported
    assert index["launches"][1]["resumed_at_optimizer_step"] == 500


def test_gym_cache_hashes(tmp_path):
    root = tmp_path / "cache"
    (root / "train" / "clip").mkdir(parents=True)
    (root / "train" / "clip" / "poses.npz").write_bytes(b"abc")
    (root / "index.json").write_text(json.dumps({"dataset": "d", "sources": {"calibration_sha256": "x"}}))
    record = run_manifest.gym_cache_record(str(root))
    assert record["dataset"] == "d" and record["sources"]["calibration_sha256"] == "x"
    assert record["targets_sha256"]["train"]["train/clip/poses.npz"] == run_manifest.file_sha256(
        root / "train" / "clip" / "poses.npz")
    assert len(record["index"]["sha256"]) == 64
