from dataclasses import dataclass

from datasets import load_dataset

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class XSTestConfig:
    name: str
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("xs_test")
class XSTestDataset(PromptDataset):
    def __init__(self, config: XSTestConfig):
        super().__init__(config)
        dataset = load_dataset("walledai/XSTest")["test"]
        dataset = [d["prompt"] for d in dataset if d["label"] == "safe"]

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = [dataset[i] for i in self.idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages[idx]
        conversation = [
            {"role": "user", "content": msg},
        ]
        return conversation
