import subprocess
from pathlib import Path
from typing import Optional

from warg import ensure_existence
from warg import Number

FORMAT_LIST = [".mp4", ".avi", ".mkv", ".flv", ".mov"]
AUDIO_FORMAT = ".aac"

__all__ = ["extract_frames"]


def extract_frames(
    file_path: Path,
    frame_out_dir: Optional[Path] = None,
    audio_out_dir: Optional[Path] = None,
    rate: Number = 25,
    frame_format: str = "jpg",
    extract_sound: bool = True,
    ffmpeg_path: Path = "ffmpeg",
) -> None:
    """

    :param file_path:
    :type file_path:
    :param frame_out_dir:
    :type frame_out_dir:
    :param audio_out_dir:
    :type audio_out_dir:
    :param rate:
    :type rate:
    :param frame_format:
    :type frame_format:
    :param extract_sound:
    :type extract_sound:
    :param ffmpeg_path:
    :type ffmpeg_path:
    """
    root_dir = file_path.parent
    if frame_out_dir is None:
        frame_out_dir = ensure_existence(root_dir / file_path.stem / "frames")
    if audio_out_dir is None:
        audio_out_dir = ensure_existence(frame_out_dir.parent / "audio")

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
                str(frame_out_dir / f"%d.{frame_format}"),
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
                    str(audio_out_dir / f"track{AUDIO_FORMAT}"),
                ]
            )
            print(f"end extracting {file_path} audio")
    else:
        print(f"{file_path} is not a video file")


if __name__ == "__main__":
    # a = Path.home() / "DataWin" / "DeepFake" / "Frontier" / "Originals" / "thomas_old_high_res.mp4"

    a = (
        Path.home()
        / "SynologyDrive"
        / "Frontier"
        / "Fra Frontier"
        / "Personer"
        / "Peter AG"
        / "Peter AG 1983+1991.mp4"
    )
    ffmpeg_path = "ffmpeg"
    if False:
        ffmpeg_path = (
            Path.home()
            / "OneDrive - Alexandra Instituttet"
            / "Applications"
            / "ffmpeg-5.0-essentials_build"
            / "bin"
            / "ffmpeg.exe"
        )
    extract_frames(
        a,
        ffmpeg_path=ffmpeg_path,
    )
