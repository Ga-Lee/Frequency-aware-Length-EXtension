import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ans import sample_ans_noise


def test_rho_minus_one_is_alternating():
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    noise = sample_ans_noise(
        num_frames=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=g,
        chunk_size=3,
        latent_shape=(2, 2, 2),
        rho=-1.0,
    )
    z0 = noise[0, 0]
    z3 = noise[0, 3]
    assert torch.allclose(noise[0, 1], -z0)
    assert torch.allclose(noise[0, 2], z0)
    assert torch.allclose(noise[0, 4], -z3)
    assert torch.allclose(noise[0, 5], z3)
    assert not torch.allclose(z0, z3)


def test_marginal_and_correlation():
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    rho = -0.5
    noise = sample_ans_noise(
        num_frames=3000,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=g,
        chunk_size=3,
        latent_shape=(1, 1, 8),
        rho=rho,
    )
    flat = noise[0].reshape(3000, -1)
    mean = flat.mean().item()
    var = flat.var(unbiased=False).item()
    assert abs(mean) < 0.05
    assert abs(var - 1.0) < 0.05
    corr = torch.mean(flat[0::3] * flat[1::3]).item()
    assert abs(corr - rho) < 0.05


def test_rejects_bad_length():
    try:
        sample_ans_noise(
            num_frames=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=None,
            chunk_size=3,
            latent_shape=(1, 1, 1),
            rho=-1.0,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_rho_minus_one_is_alternating()
    test_marginal_and_correlation()
    test_rejects_bad_length()
    print("ans tests passed")
