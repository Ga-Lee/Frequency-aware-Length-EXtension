"""FLEX inference on Self Forcing (Wan2.1-T2V-1.3B, DMD few-step checkpoint)."""
import argparse
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from pipeline import CausalInferencePipeline
from utils.ans import sample_ans_noise
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.inference_runtime import prompt_indices, sample_seed_for

ROOT = Path(__file__).resolve().parent
LATENT_SHAPE = (16, 60, 104)  # 832x480 for Wan2.1-T2V-1.3B


def existing_file(path: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    rooted = ROOT / path
    if rooted.is_file():
        return str(rooted)
    return str(candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="FLEX long-video inference on Self Forcing")
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_flex.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--wan_model_root", type=str, default=None,
                        help="Wan2.1-T2V-1.3B directory. Defaults to WAN_MODEL_ROOT or wan_models/Wan2.1-T2V-1.3B")
    parser.add_argument("--data_path", type=str, required=True, help="Text file with one prompt per line")
    parser.add_argument("--extended_prompt_path", type=str, default=None,
                        help="Optional conditioning prompts, one per line, aligned with --data_path")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_output_frames", type=int, default=126,
                        help="Latent frames. 21 ~ 5s, 126 ~ 30s, 240 ~ 60s. Must be divisible by 3.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--noise_rho", type=float, default=None,
                        help="ANS correlation in [-1, 1]. Default is the config value, -1.")
    parser.add_argument("--rope_extrapolation", type=str, default=None, choices=["none", "rope_modulation"])
    parser.add_argument("--rope_clamp_alpha", type=float, default=None)
    parser.add_argument("--rope_clamp_beta", type=float, default=None)
    parser.add_argument("--local_attn_size", type=int, default=None)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Name files by prompt index instead of prompt text")
    return parser.parse_args()


def load_generator_state(checkpoint_path: str, use_ema: bool):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preferred_key = "generator_ema" if use_ema else "generator"
    if isinstance(state_dict, dict):
        if preferred_key in state_dict:
            sd = state_dict[preferred_key]
        elif "generator" in state_dict:
            sd = state_dict["generator"]
        elif "model" in state_dict:
            sd = state_dict["model"]
        else:
            sd = state_dict
    else:
        sd = state_dict

    def strip_prefix(name: str) -> str:
        return (
            name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
            .replace("module.", "")
        )

    return {strip_prefix(key): value for key, value in sd.items()}


def safe_filename(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return (cleaned[:limit] or "prompt")


def make_noise(num_frames, device, generator, chunk_size, rho):
    return sample_ans_noise(
        num_frames=num_frames,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        chunk_size=chunk_size,
        latent_shape=LATENT_SHAPE,
        rho=rho,
    )


def save_video(video, output_path):
    frames = rearrange(video, "b t c h w -> b t h w c")[0].cpu()
    frames = (255.0 * frames).clamp(0, 255).to(torch.uint8)
    write_video(output_path, frames, fps=16)


def main():
    args = parse_args()
    config = OmegaConf.load(existing_file(args.config_path))
    if "model_kwargs" not in config:
        config.model_kwargs = {}

    config.wan_model_root = args.wan_model_root
    config.model_kwargs.rope_target_frames = int(args.num_output_frames)
    if args.rope_extrapolation is not None:
        config.rope_extrapolation = args.rope_extrapolation
    if args.rope_clamp_alpha is not None:
        config.rope_clamp_alpha = args.rope_clamp_alpha
    if args.rope_clamp_beta is not None:
        config.rope_clamp_beta = args.rope_clamp_beta
    if args.local_attn_size is not None:
        config.model_kwargs.local_attn_size = args.local_attn_size
    if args.sink_size is not None:
        config.model_kwargs.sink_size = args.sink_size
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    num_samples = int(config.get("num_samples", 1))
    noise_rho = float(config.noise_rho if args.noise_rho is None else args.noise_rho)
    chunk_size = int(config.num_frame_per_block)
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if args.num_output_frames <= 0 or args.num_output_frames % chunk_size != 0:
        raise ValueError(
            f"--num_output_frames ({args.num_output_frames}) must be divisible by "
            f"num_frame_per_block ({chunk_size}) and must be positive."
        )
    if not -1.0 <= noise_rho <= 1.0:
        raise ValueError(f"noise_rho must be in [-1, 1], got {noise_rho}")

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device("cuda")
        rank, world_size = 0, 1
    set_seed(args.seed)

    extended_prompt_path = (
        existing_file(args.extended_prompt_path) if args.extended_prompt_path else None
    )
    dataset = TextDataset(existing_file(args.data_path), extended_prompt_path)
    if len(dataset) == 0:
        raise ValueError("The prompt file must contain at least one non-empty prompt")
    sampler = prompt_indices(len(dataset), rank, world_size)
    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    # Ranks with no prompts exit before loading models. No later collective is needed.
    if len(sampler) == 0:
        dist.destroy_process_group()
        return

    low_memory = args.low_memory or get_cuda_free_memory_gb(device) < 40
    if rank == 0:
        train_frames = int(config.rope_train_frames)
        scale = max(1.0, args.num_output_frames / train_frames)
        print(
            f"[FLEX] frames={args.num_output_frames} S={scale:.4f} "
            f"rope={config.rope_extrapolation} alpha={config.rope_clamp_alpha} beta={config.rope_clamp_beta} "
            f"rho={noise_rho} local_attn_size={config.model_kwargs.local_attn_size} "
            f"sink_size={config.model_kwargs.sink_size} ema={args.use_ema}"
        )

    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = load_generator_state(args.checkpoint_path, args.use_ema)
    pipeline.generator.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    for batch in tqdm(dataloader, disable=(rank != 0)):
        idx = int(batch["idx"].item())
        prompt = batch["prompts"][0]
        if "extended_prompts" in batch:
            cond_prompt = batch["extended_prompts"][0]
        else:
            cond_prompt = prompt

        for sample_idx in range(num_samples):
            sample_seed = sample_seed_for(args.seed, idx, sample_idx, num_samples)
            generator = torch.Generator(device=device)
            generator.manual_seed(sample_seed)
            noise = make_noise(args.num_output_frames, device, generator, chunk_size, noise_rho)
            with torch.inference_mode():
                video = pipeline.inference(
                    noise=noise,
                    text_prompts=[cond_prompt],
                    return_latents=False,
                    low_memory=low_memory,
                    generator=generator,
                )
            if args.save_with_index:
                name = f"{idx:04d}-{sample_idx}.mp4"
            else:
                name = f"{idx:04d}_{safe_filename(prompt)}-{sample_idx}.mp4"
            save_video(video, os.path.join(args.output_folder, name))
            pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
