import os
import subprocess
from pathlib import Path

from apppath import ensure_existence


def spilt_video(
    video_path: Path,
    start_time="01:40:00",
    stop_time="01:50:00",
    split_dir: Path = None,
    ffmpeg_path: Path = "ffmpeg",
) -> None:
    if split_dir is None:
        split_dir = ensure_existence(video_path.parent / "split")

    if not split_dir.is_dir():
        ensure_existence(split_dir)

    ext = video_path.suffix
    name = video_path.stem

    while os.path.exists(str(split_dir / (name + ext))):
        name = name + "-1"

    subprocess.call(
        [
            ffmpeg_path,
            "-ss",
            start_time,
            "-to",
            stop_time,
            "-accurate_seek",
            "-i",
            video_path,
            "-vcodec",
            "copy",
            "-acodec",
            "copy",
            "-avoid_negative_ts",
            "1",
            str(split_dir / (name + ext)),
            "-y",
        ]
    )


if __name__ == "__main__":
    spilt_video(
        Path.home() / "Downloads" / "brandt.mp4",
        ffmpeg_path=Path.home()
        / "Downloads"
        / "ffmpeg-5.0-essentials_build"
        / "ffmpeg-5.0-essentials_build"
        / "bin"
        / "ffmpeg.exe",
    )
