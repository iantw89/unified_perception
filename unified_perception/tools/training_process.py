"""Start, stop, or inspect a detached, checkpoint-resumable training process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from ..config import load_config


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs") / "gym_sixcam_yolov8x_c7.yaml"


def running(process: psutil.Process) -> bool:
    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE


def read_record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, int):
        return {"pid": value}
    if not isinstance(value, dict) or not isinstance(value.get("pid"), int):
        raise ValueError(f"Invalid training PID record: {path}")
    return value


def is_trainer(process: psutil.Process, config: Path) -> bool:
    command = process.cmdline()
    module = "unified_perception.tools.train"
    if module not in command or "--config" not in command:
        return False
    index = command.index("--config") + 1
    if index >= len(command):
        return False
    path = Path(command[index])
    if not path.is_absolute():
        path = Path(process.cwd()) / path
    return path.resolve() == config.resolve()


def recorded_process(path: Path, config: Path) -> psutil.Process | None:
    if not path.exists():
        return None
    record = read_record(path)
    try:
        process = psutil.Process(record["pid"])
        if not running(process):
            return None
        if "create_time" in record:
            if abs(process.create_time() - record["create_time"]) > 0.01:
                return None  # The PID has been reused; never signal the new owner.
            if Path(record["config"]).resolve() != config.resolve() or not is_trainer(process, config):
                raise RuntimeError(f"PID {process.pid} does not match the recorded training command")
        else:
            # Old PowerShell launchers stored a cmd.exe parent PID, not the Python PID.
            candidates = [process, *process.children(recursive=True)]
            if not any(is_trainer(child, config) for child in candidates if running(child)):
                raise RuntimeError(f"Legacy PID {process.pid} cannot be verified as this training run")
        return process
    except psutil.NoSuchProcess:
        return None


def launch_options(platform: str) -> dict:
    if platform == "win32":
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_tree(process: psutil.Process, timeout: float = 10) -> None:
    """Freeze the parent before discovering workers so it cannot keep spawning them."""
    try:
        process.suspend()
    except psutil.NoSuchProcess:
        return
    processes = [process]
    try:
        processes = [*process.children(recursive=True), process]
        for child in processes[:-1]:
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
        for child in reversed(processes[:-1]):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs(processes, timeout=timeout)
        alive = [child.pid for child in alive if running(child)]
        if alive:
            raise RuntimeError(f"Training processes did not exit: {alive}")
    except BaseException:
        for child in processes:
            try:
                child.resume()
            except psutil.NoSuchProcess:
                pass
        raise


def start_training(config: Path, run: Path) -> int:
    run.mkdir(parents=True, exist_ok=True)
    pid_file = run / "train.pid"
    existing = recorded_process(pid_file, config)
    if existing is not None:
        raise RuntimeError(f"Run already active (PID {existing.pid}); stop it first")
    if pid_file.exists():
        pid_file.unlink()
    # Exclusive creation prevents simultaneous launchers from starting duplicate trainers.
    with pid_file.open("x", encoding="utf-8") as pid_stream:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                               "garbage_collection_threshold:0.8,max_split_size_mb:256")
        command = [sys.executable, "-u", "-m", "unified_perception.tools.train",
                   "--config", str(config), "--resume"]
        child = None
        try:
            with (run / "console.log").open("ab", buffering=0) as log:
                stamp = datetime.now(timezone.utc).isoformat()
                log.write(f"\n===== {stamp} start/resume {config} =====\n".encode("utf-8"))
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                         **launch_options(sys.platform))
            process = psutil.Process(child.pid)
            json.dump({"pid": child.pid, "create_time": process.create_time(),
                       "config": str(config)}, pid_stream)
            pid_stream.flush()
            time.sleep(0.5)
            if child.poll() is not None:
                raise RuntimeError(f"Trainer exited with code {child.returncode}; see {run / 'console.log'}")
        except BaseException:
            if child is not None:
                try:
                    stop_tree(psutil.Process(child.pid))
                except psutil.NoSuchProcess:
                    pass
            # Windows cannot remove a file while its own write handle is open.
            pid_stream.close()
            pid_file.unlink()
            raise
    print(f"Started PID {child.pid}; log {run / 'console.log'}", flush=True)
    return child.pid


def stop_training(config: Path, run: Path) -> None:
    pid_file = run / "train.pid"
    if not pid_file.exists():
        raise FileNotFoundError(f"No training PID record: {pid_file}")
    process = recorded_process(pid_file, config)
    if process is None:
        print("Recorded training process is no longer running")
    else:
        stop_tree(process)
        print(f"Stopped PID {process.pid} and its workers")
    pid_file.unlink()
    checkpoint = run / "last.pt"
    if checkpoint.exists():
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        print(f"Resume point: optimizer step {state['optimizer_steps']} ({checkpoint})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = args.config if args.config.is_absolute() else REPO / args.config
    config = config.resolve()
    # Training configs contain repository-relative dataset paths.
    os.chdir(REPO)
    cfg = load_config(str(config))
    run = Path(cfg.train.output_dir).resolve()
    if args.action == "start":
        start_training(config, run)
    elif args.action == "stop":
        stop_training(config, run)
    else:
        process = recorded_process(run / "train.pid", config)
        print(f"Running (PID {process.pid})" if process else "Not running")


if __name__ == "__main__":
    main()
