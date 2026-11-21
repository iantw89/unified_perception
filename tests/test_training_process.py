from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import psutil
import pytest

from unified_perception.tools import training_process as manager


class Process:
    def __init__(self, pid=42, config=None, created=100):
        self.pid = pid
        self.config = config
        self.created = created
        self.alive = True
        self.workers = []
        self.events = []

    def is_running(self):
        return self.alive

    def status(self):
        return psutil.STATUS_RUNNING

    def create_time(self):
        return self.created

    def cmdline(self):
        return (["python", "-m", "unified_perception.tools.train", "--config", str(self.config)]
                if self.config else ["unrelated"])

    def cwd(self):
        return str(Path.cwd())

    def children(self, recursive=False):
        return self.workers

    def suspend(self):
        self.events.append("suspend")

    def kill(self):
        self.events.append("kill")
        self.alive = False


def test_launch_options_are_platform_specific():
    assert manager.launch_options("linux") == {"start_new_session": True}
    assert manager.launch_options("darwin") == {"start_new_session": True}
    if sys.platform == "win32":
        options = manager.launch_options("win32")
        assert options["creationflags"] & subprocess.DETACHED_PROCESS
        assert options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP


def test_pid_record_and_reuse_guard(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    path = tmp_path / "train.pid"
    path.write_text(json.dumps({"pid": 42, "create_time": 100, "config": str(config)}))
    process = Process(config=config)
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: process)
    assert manager.recorded_process(path, config) is process
    process.created = 200
    assert manager.recorded_process(path, config) is None
    assert not process.events


def test_legacy_parent_pid_requires_matching_trainer(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    path = tmp_path / "train.pid"
    path.write_text("42\n", encoding="utf-8-sig")
    parent = Process()
    parent.workers = [Process(43, config)]
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: parent)
    assert manager.recorded_process(path, config) is parent
    parent.workers = [Process(43, tmp_path / "other.yaml")]
    with pytest.raises(RuntimeError, match="cannot be verified"):
        manager.recorded_process(path, config)
    assert not parent.events


def test_missing_dead_and_invalid_pid_records(tmp_path, monkeypatch):
    path = tmp_path / "train.pid"
    config = tmp_path / "config.yaml"
    assert manager.recorded_process(path, config) is None
    path.write_text('{"pid": "bad"}')
    with pytest.raises(ValueError, match="Invalid"):
        manager.read_record(path)
    path.write_text("42")
    process = Process()
    process.alive = False
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: process)
    assert manager.recorded_process(path, config) is None


def test_stop_tree_freezes_then_kills_workers(monkeypatch):
    parent, child = Mock(spec=psutil.Process), Mock(spec=psutil.Process)
    parent.children.return_value = [child]
    monkeypatch.setattr(manager.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    manager.stop_tree(parent)
    parent.suspend.assert_called_once()
    parent.kill.assert_called_once()
    child.suspend.assert_called_once()
    child.kill.assert_called_once()


def test_failed_stop_does_not_leave_parent_suspended():
    parent = Mock(spec=psutil.Process)
    parent.children.side_effect = psutil.AccessDenied(42)
    with pytest.raises(psutil.AccessDenied):
        manager.stop_tree(parent)
    parent.resume.assert_called_once()
    parent.kill.assert_not_called()


def test_start_uses_python_without_shell_and_records_identity(tmp_path, monkeypatch):
    config = tmp_path / "config with spaces.yaml"
    config.touch()
    process = Process(config=config)
    calls = []

    class Child:
        pid = 42

        def poll(self):
            return None

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["stderr"] == subprocess.STDOUT
        return Child()

    monkeypatch.setattr(manager.subprocess, "Popen", popen)
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(manager.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("FFMPEG_SHARED_BIN", "user-selected-path")
    run = tmp_path / "run"
    assert manager.start_training(config, run) == 42
    command, options = calls[0]
    assert command == [sys.executable, "-u", "-m", "unified_perception.tools.train",
                       "--config", str(config), "--resume"]
    assert "shell" not in options
    assert options["env"]["FFMPEG_SHARED_BIN"] == "user-selected-path"
    assert options["env"]["PYTHONIOENCODING"] == "utf-8"
    assert manager.read_record(run / "train.pid")["create_time"] == 100
    with pytest.raises(RuntimeError, match="already active"):
        manager.start_training(config, run)
    assert len(calls) == 1


def test_failed_launch_removes_pid_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("launch failed")

    monkeypatch.setattr(manager.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="launch failed"):
        manager.start_training(tmp_path / "config.yaml", tmp_path / "run")
    assert not (tmp_path / "run" / "train.pid").exists()


def test_immediate_trainer_exit_cleans_up(tmp_path, monkeypatch):
    class Child:
        pid = 42
        returncode = 1

        def poll(self):
            return 1

    process = Process()
    monkeypatch.setattr(manager.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(manager, "stop_tree", lambda child: None)
    monkeypatch.setattr(manager.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Trainer exited"):
        manager.start_training(tmp_path / "config.yaml", tmp_path / "run")
    assert not (tmp_path / "run" / "train.pid").exists()


def test_stop_dead_record_reports_checkpoint(tmp_path, monkeypatch, capsys):
    import torch

    (tmp_path / "train.pid").write_text("42")
    torch.save({"optimizer_steps": 99}, tmp_path / "last.pt")
    monkeypatch.setattr(manager, "recorded_process", lambda path, config: None)
    manager.stop_training(tmp_path / "config.yaml", tmp_path)
    assert not (tmp_path / "train.pid").exists()
    assert "optimizer step 99" in capsys.readouterr().out


def test_real_process_tree_stop(tmp_path):
    """Exercise actual process-tree termination with CPU-only, task-owned helpers."""
    child_pid = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"Path({str(child_pid)!r}).write_text(str(child.pid)); time.sleep(60)"
    )
    helper = subprocess.Popen([sys.executable, "-c", code])
    parent = psutil.Process(helper.pid)
    try:
        import time

        deadline = time.monotonic() + 10
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid.exists()
        worker = psutil.Process(int(child_pid.read_text()))
        manager.stop_tree(parent)
        helper.wait(timeout=5)
        assert not manager.running(parent)
        assert not manager.running(worker)
    finally:
        if helper.poll() is None:
            manager.stop_tree(parent)
            helper.wait(timeout=5)
