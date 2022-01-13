#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

from typing import Sequence

import numpy

from apppath import ensure_existence
from draugr.numpy_utilities.datasets.splitting import SplitEnum
from draugr.numpy_utilities.signal_utilities.signal_statistics import root_mean_square
from draugr.numpy_utilities.signal_utilities.truncation import min_length_truncate_batch

LOG_MAGNITUDE_MULTIPLIER = 20
LOG_POWER_MULTIPLIER = 10

__all__ = ["mix_ratio"]


def mix_ratio(
    s1: Sequence,
    s2: Sequence,
    db_ratio: float = 0,
    *,
    log_multiplier: int = LOG_MAGNITUDE_MULTIPLIER,
) -> Sequence:
    """
    # Function to mix clean speech and noise at various SNR levels

    # Normalizing to -25 dB FS

    :param s1:
    :param s2:
    :param db_ratio:
    :param log_multiplier:
    :return:"""
    s1, s2 = min_length_truncate_batch((s1, s2))
    s1_max = numpy.abs(s1).max()
    mix = (s1 / s1_max) + (
        (s2 / numpy.abs(s2).max()) * (root_mean_square(s1) / root_mean_square(s2))
    ) / (10 ** (db_ratio / log_multiplier))
    return (mix / numpy.abs(mix).max()) * s1_max


if __name__ == "__main__":

    def asad() -> None:
        """
        :rtype: None
        """
        from neodroidaudition.data.recognition.libri_speech import LibriSpeech
        from neodroidaudition.noise_generation.gaussian_noise import white_noise

        from pathlib import Path

        libri_speech = LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech"
        )
        files, sr = zip(*[(v[0].numpy(), v[1]) for _, v in zip(range(1), libri_speech)])
        assert all([sr[0] == s for s in sr[1:]])

        mixed = mix_ratio(files[0], white_noise(files[0].shape[-1]))
        print(mixed)

    def asadsa() -> None:
        """
        :rtype: None
        """
        from draugr.torch_utilities import to_tensor
        from neodroidaudition.data.recognition.libri_speech import LibriSpeech
        from neodroidaudition.noise_generation.gaussian_noise import white_noise
        import torchaudio

        from pathlib import Path

        libri_speech = LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech",
            split=SplitEnum.testing,
        )
        files, sr = zip(
            *[(v[0].numpy(), v[1]) for _, v in zip(range(20), libri_speech)]
        )
        assert all([sr[0] == s for s in sr[1:]])

        mix = files[0]
        for file in files[1:]:
            mix = mix_ratio(mix, file, 0)

        torchaudio.save(
            str(ensure_existence(Path.cwd() / "exclude") / f"mixed_even_babble.wav"),
            to_tensor(mix),
            int(sr[0]),
        )

        for ratio in range(-20, 20 + 1, 5):
            torchaudio.save(
                str(ensure_existence(Path.cwd() / "exclude") / f"mixed_{ratio}.wav"),
                to_tensor(mix_ratio(files[0], files[-1], ratio)),
                int(sr[0]),
            )

    def asadsa2() -> None:
        """
        :rtype: None
        """
        from draugr.torch_utilities import to_tensor
        from neodroidaudition.data.recognition.libri_speech import LibriSpeech
        from neodroidaudition.noise_generation.gaussian_noise import white_noise
        import torchaudio
        from pathlib import Path

        libri_speech = LibriSpeech(
            path=Path.home() / "Data" / "Audio" / "Speech" / "LibriSpeech"
        )
        files, sr = zip(*[(v[0].numpy(), v[1]) for _, v in zip(range(1), libri_speech)])
        assert all([sr[0] == s for s in sr[1:]])

        normed = files[0]
        mixed = mix_ratio(normed, normed, 0)
        mixed2 = mix_ratio(mixed, mixed, 0)
        print(normed, mixed)
        print(mixed2, mixed)
        print(root_mean_square(normed))
        print(root_mean_square(mixed))
        print(root_mean_square(mixed2))
        assert numpy.allclose(normed, mixed)
        assert numpy.allclose(mixed2, mixed)
        torchaudio.save(
            str(ensure_existence(Path.cwd() / "exclude") / "mixed_same.wav"),
            to_tensor(mixed),
            int(sr[0]),
        )

    # asad()
    asadsa()
    # asadsa2()
