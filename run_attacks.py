import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from llm_quick_check.run_attacks import run_attacks_main, print_exceptions

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.recompile_limit = 512  # needed for gemma 3 on AutoDAN


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def hydra_main(cfg: DictConfig) -> None:
    run_attacks_main(cfg)

if __name__ == "__main__":
    hydra_main()
