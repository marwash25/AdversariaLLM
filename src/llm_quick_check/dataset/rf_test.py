from dataclasses import dataclass

from datasets import load_dataset

from llm_quick_check.types import Conversation

from .prompt_dataset import BaseDataConfig, PromptDataset


@dataclass
class RFTestConfig(BaseDataConfig):
    name: str = "rf_test"
    path: str | None = None


@PromptDataset.register("rf_test")
class RFTestDataset(PromptDataset):
    def __init__(self, config: RFTestConfig):
        super().__init__(config)
        _path = config.path if config.path else "redflag-tokens/harmful-harmless-eval"
        dataset = load_dataset(_path, "harmful")['test']
        messages = [d["Behavior"] for d in dataset]
        targets = [d["Prefilling"] for d in dataset]

        # shuffle and select
        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = [messages[i] for i in self.idx]
        self.targets = [targets[i] for i in self.idx]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": self.targets[idx]},
        ]
        return conversation

