"""Log-mel spectrogram frontend, pure ``torch`` — no ``torchaudio`` dependency."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Return a ``[n_mels, n_fft // 2 + 1]`` triangular mel filterbank matrix."""

    f_max = float(f_max if f_max is not None else sample_rate / 2)
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sample_rate / 2, n_freqs)

    mel_min = _hz_to_mel(torch.tensor(f_min))
    mel_max = _hz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(float(mel_min), float(mel_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filterbank = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (freqs - left) / (center - left).clamp_min(1e-10)
        falling = (right - freqs) / (right - center).clamp_min(1e-10)
        filterbank[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return filterbank


class LogMelSpectrogram(nn.Module):
    """Waveform ``[batch, samples]`` -> log-mel spectrogram ``[batch, n_mels, frames]``.

    A fixed (non-trainable), pure-``torch`` STFT + triangular mel filterbank —
    deliberately dependency-free rather than requiring ``torchaudio``.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        if n_mels <= 0 or n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_mels, n_fft, and hop_length must be positive")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "filterbank",
            build_mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must be [batch, samples]")
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.dtype),
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2.0)
        mel = torch.einsum("mf,bft->bmt", self.filterbank.to(waveform.dtype), power)
        return torch.log(mel.clamp_min(1e-10))

    def frame_count(self, num_samples: int) -> int:
        return num_samples // self.hop_length + 1
