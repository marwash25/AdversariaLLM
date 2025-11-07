import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import hydra
import torch
from omegaconf import DictConfig

from llm_quick_check.run_judges import run_judges_main

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_judges_main(cfg)


if __name__ == "__main__":
    main()
