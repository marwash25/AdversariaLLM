import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from run_judges import run_judges
from src.attacks import Attack, AttackResult
from src.dataset import PromptDataset
from src.errors import print_exceptions
from src.io_utils import (RunConfig, filter_config, free_vram, load_model_and_tokenizer,
                          log_attack, setup_hf_cache_on_slurm_tmpdir)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.recompile_limit = 512  # needed for gemma 3 on AutoDAN


def select_configs(cfg: DictConfig, name: str | ListConfig | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        if isinstance(name, ListConfig):
            return [(n, cfg[n]) for n in name]
        return [(name, cfg[name])]
    return list(cfg.items())


def collect_configs(cfg: DictConfig) -> list[RunConfig]:
    models_to_run = select_configs(cfg.models, cfg.model)
    datasets_to_run = select_configs(cfg.datasets, cfg.dataset)
    attacks_to_run = select_configs(cfg.attacks, cfg.attack)

    all_run_configs = []
    for model, model_params in models_to_run:
        for dataset, dataset_params in datasets_to_run:
            temp_dataset = PromptDataset.from_name(dataset)(dataset_params)
            dset_len = len(temp_dataset)
            dataset_params["idx"] = temp_dataset.config_idx  ## TODO: maybe set this to temp_dataset.idx.to_list(), but need to handle the batching properly 
            for attack, attack_params in attacks_to_run:
                run_config = RunConfig(
                    model,
                    dataset,
                    attack,
                    model_params,
                    dataset_params,
                    attack_params,
                )
                run_config = filter_config(run_config, dset_len, overwrite=cfg.overwrite, use_database=cfg.get("use_database", False))
                if run_config is not None:
                    all_run_configs.append(run_config)
    return all_run_configs


def run_attacks(all_run_configs: list[RunConfig], cfg: DictConfig, date_time_string: str, save_format: str = "default") -> None:
    last_model = None
    last_dataset = None
    last_attack = None
    for run_config in all_run_configs:
        # To avoid reloading the model and dataset for every attack,
        # we only reload something if it's different from the last run
        if last_model != run_config.model:
            logging.info(f"Target: {run_config.model}\n{OmegaConf.to_yaml(run_config.model_params, resolve=True)}")
            last_model = run_config.model
            model, tokenizer = load_model_and_tokenizer(run_config.model_params)
        if last_dataset != run_config.dataset:
            logging.info(f"Dataset: {run_config.dataset}\n{OmegaConf.to_yaml(run_config.dataset_params, resolve=True)}")
            last_dataset = run_config.dataset
            dataset = PromptDataset.from_name(run_config.dataset)(run_config.dataset_params)
            if run_config.dataset_params["idx"] != dataset.idx.tolist():
                logging.info(f"Dataset {run_config.dataset} has changed indices from {run_config.dataset_params['idx']} to {dataset.idx.tolist()}")
                run_config.dataset_params["idx"] = dataset.idx.tolist()
        if last_attack != run_config.attack:
            logging.info(f"Attack: {run_config.attack}\n{OmegaConf.to_yaml(run_config.attack_params, resolve=True)}")
            last_attack = run_config.attack

        attack: Attack[AttackResult] = Attack.from_name(run_config.attack)(run_config.attack_params)
        results = attack.run(model, tokenizer, dataset)  # type: ignore

        log_attack(run_config, results, cfg, date_time_string=date_time_string, save_format=save_format)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    date_time_string = datetime.now().strftime("%Y-%m-%d/%Hh%Mm%Ss")
    date_string, time_string = date_time_string.split('/')
    save_format = getattr(cfg, "save_format", "default")
    logging.info("-------------------")
    logging.info(f"Commencing run at `{date_time_string}`")
    logging.info(f"Saving to: {cfg.save_dir}; Save format: {save_format}")
    logging.info("-------------------")

    # 1. Parse/Collect configs for the judge and the attacks
    OmegaConf.set_struct(cfg, False)
    # Remove classifiers from config to avoid saving to mongodb
    # This way we don't re-run the attacks when only the classifiers change
    judges_to_run = cfg.pop('classifiers')
    OmegaConf.set_struct(cfg, True)
    all_run_configs = collect_configs(cfg)

    if slurm_copy_hf_cache := cfg.get("slurm_copy_hf_cache", None):
        logging.info(f"Copying models: {OmegaConf.to_yaml(slurm_copy_hf_cache)}")
        setup_hf_cache_on_slurm_tmpdir(
            source_hf_home=os.environ.get("HF_HOME"),
            model_names=slurm_copy_hf_cache.get('models'),
            dataset_names=slurm_copy_hf_cache.get('datasets'),
            set_env_vars=True,
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
        )
        logging.info(f"HF_HOME: {os.environ.get('HF_HOME')}")

    # 2. Run the attacks
    run_attacks(all_run_configs, cfg, date_time_string=date_time_string, save_format=save_format)

    # 3. Run the judges
    if judges_to_run is None:
        return
    for judge in judges_to_run:
        # Create judge config
        # TODO: not verified yet
        judge_cfg = OmegaConf.create({
            "classifier": judge,
            "suffixes": [time_string],  # Use the timestamp from this run to make sure we only judge this run
            "filter_by": None,
            "save_dir": cfg.save_dir,
            "use_database": cfg.use_database,
        })
        free_vram()
        # Run the judge
        run_judges(judge_cfg)

if __name__ == "__main__":
    main()
