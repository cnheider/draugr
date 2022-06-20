import os
import subprocess
from pathlib import Path
from typing import Optional

from apppath import ensure_existence
from draugr.ffmpeg_utilities.extract import AUDIO_FORMAT
from warg import Number, identity

__all__ = ["merge_video"]


def get_frame_format(frames_dir) -> str:
    """

    :param frames_dir:
    :type frames_dir:
    :return:
    :rtype:
    """
    for file_ in os.listdir(frames_dir):
        if os.path.splitext(file_)[-1].lower() in [".jpg", ".png"]:
            return os.path.splitext(file_)[-1].lower()


def merge_video(
    frames_dir: Path,
    merge_audio: bool = True,
    audio_dir: Optional[Path] = None,
    merge_dir: Optional[Path] = None,
    merge_rate: Number = 25,
    ffmpeg_path: Path = "ffmpeg",
):
    """

    :param frames_dir:
    :type frames_dir:
    :param merge_audio:
    :type merge_audio:
    :param audio_dir:
    :type audio_dir:
    :param merge_dir:
    :type merge_dir:
    :param merge_rate:
    :type merge_rate:
    :param ffmpeg_path:
    :type ffmpeg_path:
    """
    postfix = ""  # "_00"
    vid_dir = frames_dir.parent

    if audio_dir is None:
        audio_dir = vid_dir / "audio"

    if merge_dir is None:
        merge_dir = ensure_existence(vid_dir / "merge", sanitisation_func=identity)

    temp_out = str(merge_dir / "temp.mp4")
    subprocess.call(
        [
            str(ffmpeg_path),
            "-r",
            str(merge_rate),
            # "-pattern_type",
            # "glob",
            "-i",
            str(frames_dir / f"%d{get_frame_format(frames_dir)}"),
            "-y",
            "-c:v",
            "libx264",
            "-vf",
            "fps=25,format=yuv420p",
            temp_out,
        ]
    )

    a = []
    shortname = frames_dir.parent.name
    if merge_audio:
        sound_dir = audio_dir / ("track" + AUDIO_FORMAT)
        if sound_dir.exists():
            a.extend(["-i", str(sound_dir)])
        else:
            print(f"No audio found in {sound_dir}")

    subprocess.call(
        [
            str(ffmpeg_path),
            "-i",
            temp_out,
            *a,
            "-vcodec",
            "copy",
            "-acodec",
            "copy",
            "-y",
            str(merge_dir / f"out.mp4"),
        ]
    )


if __name__ == "__main__":
    merge_video(
        Path.home()
        / "DataWin"
        / "DeepFake"
        / "Frontier"
        / "Originals"
        / "thomas_old_high_res"
        / "frames",
        ffmpeg_path=Path.home()
        / "OneDrive - Alexandra Instituttet"
        / "Applications"
        / "ffmpeg-5.0-essentials_build"
        / "bin"
        / "ffmpeg.exe",
    )
