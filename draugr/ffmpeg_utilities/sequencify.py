from pathlib import Path

__all__ = ["sequence_files"]


def sequence_files(frames_dir: Path) -> None:
    """

    :param frames_dir:
    :type frames_dir:
    """
    for i, f in enumerate(sorted(frames_dir.iterdir(), key=lambda x: x.name)):
        if f.is_file():
            suffix = {f.suffix}
            # suffix = ".jpg"
            f.rename(frames_dir / f"{i:05d}{suffix}")


if __name__ == "__main__":
    sequence_files(
        Path(r"G:/") / "Mit drev" / "upscaled" / "restored_imgs",
    )
