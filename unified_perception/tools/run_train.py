"""Train launcher: ensures shared-FFmpeg DLLs are findable by torchcodec in all processes.

Sets FFMPEG_SHARED_BIN in the environment so DataLoader worker_init_fn can call
os.add_dll_directory() in each spawned worker on Windows.
"""
import os, sys
from train import main

FFMPEG_SHARED = r"C:\Users\iantw\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build-shared\bin"
if os.path.isdir(FFMPEG_SHARED):
    # Main process: add to DLL search path directly.
    os.add_dll_directory(FFMPEG_SHARED)
    # Also set PATH so shutil.which("ffmpeg") finds the shared build first.
    os.environ["PATH"] = FFMPEG_SHARED + os.pathsep + os.environ.get("PATH", "")
    # Export for worker_init_fn to pick up in child processes.
    os.environ["FFMPEG_SHARED_BIN"] = FFMPEG_SHARED

# Now import and run training
if __name__ == '__main__':
    main()