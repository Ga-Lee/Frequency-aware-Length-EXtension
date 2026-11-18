"""Prompt assignment and random seeds shared by single- and multi-GPU inference."""


def prompt_indices(dataset_size: int, rank: int, world_size: int) -> range:
    """Shard prompts without padding, including when there are more GPUs than prompts."""
    if dataset_size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset size, rank, or world size")
    return range(rank, dataset_size, world_size)


def sample_seed_for(base_seed: int, prompt_index: int, sample_index: int, num_samples: int) -> int:
    """Keep a sample's random stream independent of rank and GPU count."""
    if prompt_index < 0 or num_samples < 1 or not 0 <= sample_index < num_samples:
        raise ValueError("Invalid prompt index, sample index, or sample count")
    return base_seed + prompt_index * num_samples + sample_index
