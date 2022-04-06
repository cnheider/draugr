import os
import subprocess
from pathlib import Path
from typing import Optional

from apppath import ensure_existence
from warg import Number

FORMAT_LIST = [".mp4", ".avi", ".mkv", ".flv", ".mov"]
AUDIO_FORMAT = ".aac"


def extract_frames(
    file_path: Path,
    frame_out_dir: Optional[Path] = None,
    audio_out_dir: Optional[Path] = None,
    rate: Number = 25,
    frame_format: str = "jpg",
    extract_sound: bool = True,
    ffmpeg_path: Path = "ffmpeg",
):

    root_dir = file_path.parent
    if frame_out_dir is None:
        frame_out_dir = ensure_existence(root_dir / "frames")
    if audio_out_dir is None:
        audio_out_dir = ensure_existence(root_dir / "audio")

    if file_path.is_file() and file_path.suffix in FORMAT_LIST:
        print(f"start extracting {file_path} frames")
        subprocess.call(
            [
                str(ffmpeg_path),
                "-i",
                file_path,
                "-r",
                str(rate),
                "-f",
                "image2",
                "-y",
                "-qscale:v",
                "2",
                str(frame_out_dir / f"{file_path.name}-%05d.{frame_format}"),
            ]
        )
        print(f"end extracting {file_path} frames")

        if extract_sound:
            print(f"start extracting {file_path} audio")
            subprocess.call(
                [
                    str(ffmpeg_path),
                    "-i",
                    file_path,
                    "-vn",
                    "-acodec",
                    "copy",
                    "-y",
                    str(audio_out_dir / f"{file_path.name}{AUDIO_FORMAT}"),
                ]
            )
            print(f"end extracting {file_path} audio")
    else:
        print(f"{file_path} is not a video file")


if __name__ == "__main__":
    extract_frames(
        Path.home() / "Downloads" / "brandt.mp4",
        ffmpeg_path=Path.home()
        / "Downloads"
        / "ffmpeg-5.0-essentials_build"
        / "ffmpeg-5.0-essentials_build"
        / "bin"
        / "ffmpeg.exe",
    )
