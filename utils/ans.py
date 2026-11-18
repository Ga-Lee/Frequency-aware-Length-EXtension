"""Antiphase noise sampling for chunk-wise autoregressive video diffusion.

Within a chunk of length f:
    z_0 ~ N(0, I)
    z_u = rho * z_{u-1} + sqrt(1 - rho^2) * eps_u
Chunks are sampled independently. Each frame stays standard normal.
rho = -1 is the alternating pattern [z, -z, z, ...] used by FLEX.
"""
import math
from typing import Optional, Tuple

import torch


def sample_ans_noise(
    *,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
    chunk_size: int,
    latent_shape: Tuple[int, int, int],
    rho: float,
) -> torch.Tensor:
    if num_frames < 0:
        raise ValueError(f"num_frames must be >= 0, got {num_frames}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if num_frames % chunk_size != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by chunk_size ({chunk_size})."
        )
    rho = float(rho)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    channels, height, width = latent_shape
    noise = torch.empty((1, num_frames, channels, height, width), device=device, dtype=dtype)
    if num_frames == 0:
        return noise

    std = math.sqrt(max(0.0, 1.0 - rho * rho))
    for start in range(0, num_frames, chunk_size):
        z_prev = torch.randn(
            (1, 1, channels, height, width), device=device, dtype=dtype, generator=generator
        )
        noise[:, start:start + 1] = z_prev
        for step in range(1, chunk_size):
            if std == 0.0:
                z_curr = z_prev * rho
            else:
                epsilon = torch.randn(
                    (1, 1, channels, height, width), device=device, dtype=dtype, generator=generator
                )
                z_curr = z_prev * rho + epsilon * std
            noise[:, start + step:start + step + 1] = z_curr
            z_prev = z_curr
    return noise
