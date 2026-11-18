"""CPU regressions for distributed assignment and inference random streams."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ans import sample_ans_noise
from utils.inference_runtime import prompt_indices, sample_seed_for


@pytest.mark.parametrize("count,world_size", [(2, 8), (11, 3), (0, 4), (9, 1), (17, 8)])
def test_each_prompt_and_output_assigned_exactly_once(count, world_size):
    assignments = [list(prompt_indices(count, rank, world_size)) for rank in range(world_size)]
    indices = [index for shard in assignments for index in shard]
    assert sorted(indices) == list(range(count))
    outputs = [f"{index:04d}-{sample}.mp4" for index in indices for sample in range(3)]
    assert len(set(outputs)) == count * 3
    # DataLoader must preserve the original dataset index, including noncontiguous shards.
    for shard in assignments:
        batches = torch.utils.data.DataLoader(list(range(count)), sampler=shard, batch_size=1)
        assert [batch.item() for batch in batches] == shard


@pytest.mark.parametrize("args", [(-1, 0, 1), (2, 0, 0), (2, -1, 2), (2, 2, 2)])
def test_invalid_shards_rejected(args):
    with pytest.raises(ValueError):
        prompt_indices(*args)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_device_resolved_at_call_time():
    with patch.object(torch.cuda, "current_device", side_effect=AssertionError("eager CUDA access")):
        memory = load_source("memory_under_test", "demo_utils/memory.py")
    stats = {"active_bytes.all.current": 2**30, "reserved_bytes.all.current": 2 * 2**30}
    with patch.object(torch.cuda, "current_device", side_effect=[1, 3]), \
         patch.object(torch.cuda, "memory_stats", return_value=stats) as memory_stats, \
         patch.object(torch.cuda, "mem_get_info", return_value=(4 * 2**30, 8 * 2**30)):
        assert memory.get_cuda_free_memory_gb() == 5
        assert memory.get_cuda_free_memory_gb() == 5
        assert memory.get_cuda_free_memory_gb(torch.device("cuda:7")) == 5
        assert [call.args[0] for call in memory_stats.call_args_list] == [1, 3, 7]


@pytest.fixture
def small_pipeline():
    # Exercise the real denoising loop, replacing only heavyweight model components.
    wrapper = ModuleType("utils.wan_wrapper")
    for name in ("WanDiffusionWrapper", "WanTextEncoder", "WanVAEWrapper"):
        setattr(wrapper, name, None)
    with patch.dict(sys.modules, {"utils.wan_wrapper": wrapper}):
        module = load_source("pipeline_under_test", "pipeline/causal_inference.py")

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(local_attn_size=12)

        def get_scheduler(self):
            return SimpleNamespace(add_noise=lambda clean, noise, timestep: clean + 0.1 * noise)

        def forward(self, noisy_image_or_video, **kwargs):
            return None, noisy_image_or_video * 0.5

    class Encoder(torch.nn.Module):
        def forward(self, text_prompts):
            return {}

    class VAE(torch.nn.Module):
        def decode_to_pixel(self, latent, **kwargs):
            return latent

    args = SimpleNamespace(denoising_step_list=[2, 1], warp_denoising_step=False,
                           num_frame_per_block=3, independent_first_frame=False, context_noise=0)
    pipeline = module.CausalInferencePipeline(args, torch.device("cpu"), Denoiser(), Encoder(), VAE())
    pipeline.num_transformer_blocks = 1
    pipeline.kv_cache1 = [{}]
    pipeline.crossattn_cache = [{"is_init": False}]
    return module, pipeline


def run_sample(pipeline, prompt_index, sample_index):
    generator = torch.Generator().manual_seed(sample_seed_for(42, prompt_index, sample_index, 2))
    noise = sample_ans_noise(num_frames=6, device=torch.device("cpu"), dtype=torch.float32,
                             generator=generator, chunk_size=3, latent_shape=(1, 1, 2), rho=-1)
    return pipeline.inference(noise, ["test"], generator=generator, return_latents=True)[1]


def test_sample_streams_independent_of_rank_order_and_global_rng(small_pipeline):
    _, pipeline = small_pipeline
    expected = {(index, sample): run_sample(pipeline, index, sample)
                for index in range(5) for sample in range(2)}
    for rank in reversed(range(8)):
        for index in prompt_indices(5, rank, 8):
            for sample in range(2):
                torch.randn(31)  # Unrelated activity must not perturb either noise stage.
                torch.testing.assert_close(run_sample(pipeline, index, sample), expected[index, sample],
                                           rtol=0, atol=0)
    assert not torch.equal(expected[0, 0], expected[0, 1])
    assert not torch.equal(expected[0, 0], expected[1, 0])


def test_low_memory_uses_inference_tensor_device(small_pipeline):
    module, pipeline = small_pipeline
    noise = torch.zeros(1, 6, 1, 1, 2)
    with patch.object(module, "get_cuda_free_memory_gb", return_value=10) as query, \
         patch.object(module, "move_model_to_device_with_memory_preservation") as move:
        pipeline.inference(noise, ["test"], low_memory=True, generator=torch.Generator().manual_seed(0))
    query.assert_called_once_with(noise.device)
    move.assert_called_once_with(pipeline.text_encoder, target_device=noise.device, preserved_memory_gb=15)
