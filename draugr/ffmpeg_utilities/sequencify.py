import os
import subprocess
from pathlib import Path
from typing import Optional

from apppath import ensure_existence
from draugr.ffmpeg_utilities.extract import FORMAT_LIST, AUDIO_FORMAT
from warg import Number, identity


def sequencify_files(frames_dir: Path):

    for i, f in enumerate(sorted(frames_dir.iterdir(), key=lambda x: x.name)):
        if f.is_file():
            suffix = {f.suffix}
            # suffix = ".jpg"
            f.rename(frames_dir / f"{i:05d}{suffix}")


if __name__ == "__main__":
    sequencify_files(
        Path(r"G:/") / "Mit drev" / "upscaled" / "restored_imgs",
    )
