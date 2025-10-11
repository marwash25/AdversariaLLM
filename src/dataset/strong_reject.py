import os
from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class StrongRejectConfig:
    name: str
    categories: list[str]
    path: str
    split: str = "train"
    seed: int = 0
    version: Literal["small", "full"] = "small"
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("strong_reject")
class StrongRejectDataset(PromptDataset):
    def __init__(self, config: StrongRejectConfig):
        super().__init__(config)
        url = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/"
        if config.version == "small":
            url += "strongreject_small_dataset.csv"
        else:
            url += "strongreject_dataset.csv"

        self.dataset = load_dataset("csv", data_files=url)[config.split]
        self.targets = [line.rstrip() for line in open(os.path.join(config.path, "targets.txt"), "r").readlines()]
        assert len(self.dataset) == len(self.targets), "Mismatched lengths"

        self.dataset = self.dataset.add_column("target", self.targets)
        self.dataset = self.dataset.to_pandas()

        self.idx, self.config_idx = self._select_idx(config, len(self.dataset))
        self.dataset = self.dataset.iloc[self.idx]

        self.dataset = self.dataset[self.dataset["category"].isin(config.categories)]
        self.messages = self.dataset["forbidden_prompt"]
        self.targets = self.dataset["target"]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": target},
        ]
        return conversation
