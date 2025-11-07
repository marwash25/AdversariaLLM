import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Type

import torch
from llm_quick_check.types import Conversation


@dataclass
class BaseDataConfig:
    """Base configuration class for all dataset configs."""
    seed: int = 0
    idx: list[int] | int | str | None = None
    batch: int | None = None
    shuffle: bool = True


class PromptDataset(torch.utils.data.Dataset):
    """Base class for prompt datasets with simple registry support."""

    _registry: ClassVar[dict[str, Type["PromptDataset"]]] = {}

    def __init__(self, config):
        self.config = config

    @classmethod
    def register(cls, name: str):
        """Decorator used by subclasses to register themselves by name."""

        def decorator(subclass: Type["PromptDataset"]) -> Type["PromptDataset"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str) -> Type["PromptDataset"]:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise ValueError(f"Unknown dataset: {name}") from exc

    def _get_selected_idx(self, idx, batch: int | None, max_dataset_size: int):
        """Handles creating a sequence of indices for selecting a sequential batch of data.

        Args:
            idx (int, list): If batching, the starting index. If not batching, a list of indices or single int.
            batch (int, optional): Number of elements to batch. Defaults to None.
            max_dataset_size (int): Maximum size of the dataset.
        """
        # handle batching inputs
        if batch is not None:
            if isinstance(idx, Sequence):
                raise ValueError("Cannot use config.batch with a sequence of indices")
            selected_idx = list(range(idx, min(idx + batch, max_dataset_size)))
            return selected_idx
        return idx

    def _select_idx(self, config, dataset_len: int):
        """Select dataset indices according to configuration metadata."""
        # Shuffle first (on the full dataset)
        if getattr(config, "shuffle", False):
            torch.manual_seed(getattr(config, "seed", 0))
            idx = torch.randperm(dataset_len)
        else:
            idx = torch.arange(dataset_len)

        # Handle batching: convert idx + batch into a range after shuffling
        batch = getattr(config, "batch", None)
        config_idx = getattr(config, "idx", None)
        
        # Normalize config_idx: Hydra/OmegaConf sometimes converts single ints to single-element lists
        # Convert [0] -> 0, but keep [0, 1] as [0, 1]
        if isinstance(config_idx, Sequence) and len(config_idx) == 1 and not isinstance(config_idx, str):
            config_idx = config_idx[0]
        
        # If batching is enabled, convert idx to a range
        if batch is not None:
            if config_idx is None:
                raise ValueError("Cannot use config.batch when config.idx is None")
            selected_idx = self._get_selected_idx(config_idx, batch, len(idx))
        else:
            selected_idx = config_idx

        # Handle string-based idx (for eval expressions)
        if isinstance(selected_idx, str):
            if selected_idx.startswith("list(range("):
                try:
                    selected_idx = eval(selected_idx, {"__builtins__": None}, {"range": range, "list": list})
                except Exception as e:
                    raise ValueError(f"Could not parse idx string: {selected_idx}\n{e}")
            else:
                raise ValueError(f"Could not parse idx string: {selected_idx}\nDoes not start with 'list(range('.")

        # Apply selection to the shuffled indices
        if isinstance(selected_idx, int):
            idx = idx[selected_idx : selected_idx + 1]
        elif isinstance(selected_idx, Sequence):
            logging.info(f"Selecting indices: {selected_idx}")
            idx = idx[selected_idx]
        elif selected_idx is None:
            logging.info(f"Selected all data: {len(idx)}")
        else:
            raise ValueError(f"Invalid idx: {selected_idx}")

        return idx, config_idx

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Conversation:
        raise NotImplementedError
