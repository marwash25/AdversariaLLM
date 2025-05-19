"""This file can be used to generate additional completions for attack runs which are
already finished.

The easiest way to implement the filtering is using the sampling.yaml config.

Example:

filter_by:
  model_name:
    - google/gemma-3-1b-it
  attack_name:
    - gcg
    - pgd

The script also accepts a `num_return_sequences` argument, which specifies how many
completions the runs should have in the end.
"""
import os
from dataclasses import dataclass
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import sys
from src.dataset import json
from src.errors import print_exceptions
from src.io_utils import load_model_and_tokenizer, get_filtered_and_grouped_paths, get_mongodb_connection, CompactJSONEncoder, free_vram
from src.lm_utils import generate_ragged_batched
from src.judges import Judge
import copy

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class RunConfig:
    model: str
    dataset: str
    attack: str
    model_params: dict
    dataset_params: dict
    attack_params: dict
    config: dict


def eval_list_expressions_in_dict(d):
    if isinstance(d, dict):
        return {k: eval_list_expressions_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [eval_list_expressions_in_dict(v) for v in d]
    elif isinstance(d, str):
        try:
            result = eval(d, {"__builtins__": {}}, {"list": list, "range": range})
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return d


@hydra.main(config_path="./conf", config_name="sampling", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logging.info("-------------------")
    logging.info(f"Commencing run `{date_time_string}`")
    logging.info("-------------------")

    filter_by = OmegaConf.to_container(cfg.filter_by)
    filter_by = eval_list_expressions_in_dict(filter_by)
    print(filter_by)
    # get all paths
    paths = get_filtered_and_grouped_paths(filter_by, None)[("all", )]
    n = 0
    pbar = tqdm(paths, file=sys.stdout)

    model, tokenizer, judge, last_judge, last_model_params = None, None, None, None, None
    for path in pbar:
        try:
            with open(path, "r") as f:
                attack_run = json.load(f)

            gen_config = attack_run["config"]["attack_params"]["generation_config"]
            n_already_generated = gen_config["num_return_sequences"]
            n_to_generate = max(cfg.num_return_sequences - n_already_generated, 0)
            if n_to_generate == 0:
                logging.info(f"Skipping {path}, it already has {n_already_generated} completions")
                continue
            attack_run["config"]["attack_params"]["generation_config"]["num_return_sequences"] = cfg.num_return_sequences
            model_params = OmegaConf.create(attack_run["config"]["model_params"])
            if model_params != last_model_params:
                pbar.set_description(f"Loading new model and tokenizer {model_params}")
                last_model_params = model_params
                del model, tokenizer
                free_vram()
                model, tokenizer = load_model_and_tokenizer(model_params)
            # lets generate however many new ones we need
            for subrun in attack_run["runs"]:
                tokens = []
                classifiers = set()
                for step in subrun["steps"]:
                    tokens.append(torch.tensor(step["model_input_tokens"]))
                    classifiers.update(list(step["scores"].keys()))
                logging.info(f"Generating {len(subrun['steps'])}x{n_to_generate} completions.")

                additional_completions = generate_ragged_batched(
                    model,
                    tokenizer,
                    tokens,
                    num_return_sequences=n_to_generate,
                    max_new_tokens=gen_config["max_new_tokens"],
                    temperature=gen_config["temperature"],
                    top_p=gen_config["top_p"],
                    top_k=gen_config["top_k"]
                )  # (n_steps, n_to_generate)

                # have to also add classifier scores for new completions
                prompt = subrun["original_prompt"]
                for classifier in classifiers:
                    if classifier != last_judge:
                        last_judge = classifier
                        del judge
                        free_vram()
                        judge = Judge.from_name(classifier)
                    modified_prompts = []
                    for completions in additional_completions:
                        for completion in completions:
                            modified_prompt = copy.deepcopy(prompt)
                            if modified_prompt[-1]["role"] == "assistant":
                                modified_prompt[-1]["content"] += completion
                            else:
                                modified_prompt.append({"role": "assistant", "content": completion})
                            modified_prompts.append(modified_prompt)
                    results = judge(modified_prompts)
                    if all(r is None for r in results):
                        continue
                    i = 0
                    for step in subrun["steps"]:
                        for k, v in results.items():
                            step["scores"][classifier][k].extend(v[i:i+n_to_generate])
                        i += n_to_generate

                for step, completions in zip(subrun["steps"], additional_completions):
                    assert len(completions) == n_to_generate
                    step["model_completions"].extend(completions)
                    assert len(step["model_completions"]) == cfg.num_return_sequences
                    n += n_to_generate

                pbar.set_description(f"{len(subrun['steps']) * n_to_generate} | {n} total")


            json.dump(attack_run, open(path, "w"), indent=2, cls=CompactJSONEncoder)
            db = get_mongodb_connection()
            collection = db.runs
            collection.update_many({"log_file": path}, {"$set": {"config.attack_params.generation_config.num_return_sequences": cfg.num_return_sequences}})
        except Exception as e:
            logging.error(f"Error in {path}. Original exception: {e}")
            raise Exception(f"Error in {path}. Original exception: {e}") from e




if __name__ == "__main__":
    main()
