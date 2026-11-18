from torch.utils.data import Dataset


class TextDataset(Dataset):
    """One prompt per line. An optional second file supplies the conditioning text."""

    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f if line.strip()]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f if line.strip()]
            if len(self.extended_prompt_list) != len(self.prompt_list):
                raise ValueError(
                    f"Prompt count mismatch: {prompt_path} has {len(self.prompt_list)} lines, "
                    f"{extended_prompt_path} has {len(self.extended_prompt_list)} lines."
                )
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {"prompts": self.prompt_list[idx], "idx": idx}
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch
