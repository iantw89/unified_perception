"""Unified Perception: a StreamPETR-style multi-view, multi-person 3D keypoint model.

Adapts the StreamPETR approach (multi-view 2D images + camera calibration -> a
transformer detector with temporal memory and Hungarian matching) to the
human-pose domain: predict the 3D keypoints of an arbitrary number of people
from calibrated multi-view images, supervised directly by mocap ground truth.
"""

__version__ = "0.1.0"


def _register_ffmpeg_dlls() -> None:
    """Windows: put the shared FFmpeg build (``FFMPEG_SHARED_BIN``) on the DLL search path.

    Must happen before torchcodec is first imported: it caches a failed library load, and
    spawned DataLoader workers import it while unpickling the dataset, *before*
    ``worker_init_fn`` runs.  Importing this package is the first thing a worker does.
    """
    import glob
    import os
    import sys
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN", "")
        if not (ffmpeg_bin and os.path.isdir(ffmpeg_bin)):
            # Fallback: the WinGet "Gyan.FFmpeg.Shared" package (newest build first).
            pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                                   "Gyan.FFmpeg.Shared*", "*shared*", "bin")
            found = sorted(glob.glob(pattern), reverse=True)
            ffmpeg_bin = found[0] if found else ""
        if ffmpeg_bin:
            os.environ["FFMPEG_SHARED_BIN"] = ffmpeg_bin  # inherited by spawned workers
            os.add_dll_directory(ffmpeg_bin)


_register_ffmpeg_dlls()
