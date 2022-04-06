import os
import subprocess
from pathlib import Path
from typing import Optional

from apppath import ensure_existence
from draugr.ffmpeg_utilities.extract import FORMAT_LIST, AUDIO_FORMAT
from warg import Number, identity


def get_short_name(vid_dir) -> str:
    for file_ in vid_dir.iterdir():
        return file_.stem.split("-")[0]
        # if os.path.splitext(file_)[-1].lower() in FORMAT_LIST:
        #    return os.path.splitext(file_)[0]


def get_frame_format(frames_dir) -> str:
    for file_ in os.listdir(frames_dir):
        if os.path.splitext(file_)[-1].lower() in [".jpg", ".png"]:
            return os.path.splitext(file_)[-1].lower()


def merge_video(
    frames_dir: Path,
    merge_audio: bool = False,
    audio_dir: Optional[Path] = None,
    merge_dir: Optional[Path] = None,
    merge_rate: Number = 25,
    ffmpeg_path: Path = "ffmpeg",
):
    postfix = ""  # "_00"
    vid_dir = frames_dir.parent
    if audio_dir is None:
        audio_dir = vid_dir / "audio"

    if merge_dir is None:
        merge_dir = ensure_existence(vid_dir / "merge", sanitisation_func=identity)

    shortname = get_short_name(frames_dir)

    # a = f"{shortname}-%05d{postfix}{get_frame_format(frames_dir)}"  # 05d is for 5 digits in ids
    # a = "brandt.mp4-%*_00.png"
    a = f"%05d{get_frame_format(frames_dir)}"
    temp_out = str(merge_dir / "temp.mp4")
    subprocess.call(
        [
            str(ffmpeg_path),
            "-r",
            str(merge_rate),
            "-i",
            str(frames_dir / a),
            "-y",
            "-c:v",
            "libx264",
            "-vf",
            "fps=25,format=yuv420p",
            temp_out,
        ]
    )

    a = []
    if merge_audio:
        sound_dir = audio_dir / (shortname + AUDIO_FORMAT)
        a.extend(["-i", str(sound_dir)])

    while (merge_dir / f"{shortname}out.mp4").exists():
        shortname = shortname + "_1"

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
            str(merge_dir / shortname),
        ]
    )


if __name__ == "__main__":
    merge_video(
        # Path.home() / "Downloads" / "brandt" / "frames",
        # Path(r"G:/") / "Mit drev" / "upscaled" / "restored_faces",
        Path(r"G:/") / "Mit drev" / "upscaled" / "restored_imgs",
        ffmpeg_path=Path.home()
        / "OneDrive - Alexandra Instituttet"
        / "Applications"
        / "ffmpeg-5.0-essentials_build"
        / "bin"
        / "ffmpeg.exe",
    )
