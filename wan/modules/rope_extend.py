import math

import torch


def extract_inv_freq_from_complex(freq_t):
    """Recover per-plane angular frequencies from a complex RoPE table."""
    z1 = freq_t[1].to(torch.complex128)
    z0 = freq_t[0].to(torch.complex128)
    inv = torch.angle(z1 * z0.conj())
    return inv.to(freq_t.real.dtype)


def rope_modulation(inv, train_len, total_len, alpha=0.1, beta=2.5):
    """Frequency-aware temporal RoPE modulation.

    Training exposure is r = L_train * theta / (2 pi). Planes with r < alpha
    are interpolated by S = total_len / L_train, planes with r > beta keep
    their original frequency, and the band between them is linearly mixed.
    Position indices are not rescaled. S <= 1 leaves the base frequencies unchanged.
    """
    train_len = max(float(train_len), 1.0)
    ratio = max(float(total_len) / train_len, 1.0)
    if ratio <= 1.0:
        return inv

    exposure = train_len * inv / (2.0 * math.pi)
    gamma = torch.clamp((exposure - alpha) / (beta - alpha), 0.0, 1.0)
    return (1.0 - gamma) * (inv / ratio) + gamma * inv


def build_rope_freqs_extended(
    freq_t,
    start_frame,
    frame_count,
    train_frame_count,
    mode="none",
    clamp_alpha=0.1,
    clamp_beta=2.5,
    total_frame_count=None,
):
    """Return complex temporal RoPE multipliers of shape [frame_count, dim]."""
    if mode not in ("none", "rope_modulation"):
        raise ValueError(
            f"Unsupported RoPE mode {mode!r}. Use 'none' or 'rope_modulation'."
        )

    device = freq_t.device
    inv = extract_inv_freq_from_complex(freq_t)
    pos = torch.arange(start_frame, start_frame + frame_count, device=device, dtype=inv.dtype)
    if mode == "rope_modulation":
        total_len = total_frame_count if total_frame_count is not None else (start_frame + frame_count)
        inv = rope_modulation(inv, train_frame_count, total_len, clamp_alpha, clamp_beta)

    angles = pos[:, None] * inv[None, :]
    return torch.polar(torch.ones_like(angles), angles).to(freq_t.dtype)
