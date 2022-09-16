import os
import subprocess
from pathlib import Path

from warg import ensure_existence

__all__ = ["split_video"]


def split_video(
    video_path: Path,
    start_time="01:40:00",  # TODO: Find sane defaults or None
    stop_time="01:50:00",
    split_dir: Path = None,
    ffmpeg_path: Path = "ffmpeg",
) -> None:
    """Splits video into frames

    :param video_path:
    :type video_path:
    :param start_time:
    :type start_time:
    :param stop_time:
    :type stop_time:
    :param split_dir:
    :type split_dir:
    :param ffmpeg_path:
    :type ffmpeg_path:
    """
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
    split_video(
        Path.home() / "DataWin" / "DeepFake" / "Frontier" / "brandt.mp4",
        ffmpeg_path=Path.home()
        / "OneDrive - Alexandra Instituttet"
        / "Applications"
        / "ffmpeg-5.0-essentials_build"
        / "bin"
        / "ffmpeg.exe",
    )
