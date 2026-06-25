"""Train launcher: prepends shared-FFmpeg bin to PATH so torchcodec can find its DLLs."""
import os, sys

FFMPEG_SHARED = r"C:\Users\iantw\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build-shared\bin"
if os.path.isdir(FFMPEG_SHARED):
    os.add_dll_directory(FFMPEG_SHARED)
    os.environ["PATH"] = FFMPEG_SHARED + ";" + os.environ.get("PATH", "")

# Now import and run training
sys.path.insert(0, os.path.dirname(__file__))
from train import main
main()