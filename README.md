# LLM QuickCheck

A comprehensive toolkit for evaluating and comparing continuous and discrete adversarial attacks on LLMs.
This repository provides a unified framework for running various attack methods, generating adversarial prompts, and evaluating model safety and robustness.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/LLM-QC/AdversariaLLM
cd AdversariaLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## ⚙️ Configuration

### Step 1: Configure Paths

Add your environment-specific paths to the configuration file `conf/paths.yaml` (see template `conf/paths.example.yaml`). The `root_dir` is required. Several dataset configs also interpolate `data_dir` (`adv_behaviors`, `refusal_direction_data`, `strong_reject`, `rf_test`). Add that key if you use those datasets.


## 🚀 Quick Start

### Running Basic Attacks

To evaluate a model with a single attack method:

```bash
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,300)" \
    attack=gcg \
    hydra.launcher.timeout_min=240
```

### Running Multiple Attacks (Sweep)

To compare multiple attack methods:

```bash
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,300)" \
    attack=gcg,pair,autodan \
    hydra.launcher.timeout_min=240
```

This will launch 900 jobs (3 attacks × 300 prompts) and run GCG, PAIR, and AutoDAN against Phi-3 on all 300 prompts.

## 🎯 Supported Attack Methods

The framework supports various adversarial attack algorithms:

- **GCG** - Greedy Coordinate Gradient attack (with various objectives, including REINFORCE)
- **PAIR** - Prompt Automatic Iterative Refinement
- **AutoDAN** - Automatic prompt generation
- **PGD** - Projected Gradient Descent (continuous in embedding and indicator-space, with & without discretization)
- **Random Search** - Baseline random optimization
- **Human Jailbreaks** - Curated human-written prompts
- **Direct** - Direct prompt testing without optimization
- **BEAST** - Gradient-free discrete optimization
- **Best-of-N** - Jailbreaking with simple string perturbations
- **DSM** - Difference-of-submodular minimization for discrete prompt optimization (`pgm` supported; `dca` experimental)


## 📊 Evaluation and Judging

### Default Judge
By default, all completions are evaluated using **StrongREJECT**. You can change this by modifying the `classifiers` attribute in your config:

```yaml
classifiers: ["strong_reject", "harmbench", "custom_judge"]
```

### Supported Judges
For a complete list of supported judges, see: [JudgeZoo](https://github.com/LLM-QC/judgezoo)

### Running Judges Separately
```bash
python run_judges.py \
    classifier=strong_reject
```
will judge all files with strong_reject which haven't been judged yet.

## 📁 Project Structure

```
AdversariaLLM/
├── src/llm_quick_check/
│   ├── attacks/                 # Attack implementations
│   │   ├── gcg.py
│   │   ├── pair.py
│   │   ├── autodan.py
│   │   ├── dsm.py
│   │   └── ...
│   ├── dataset/                 # Dataset handling
│   ├── io_utils/                # I/O utilities
│   ├── lm_utils/                # Language model utilities
│   └── types.py                 # Type definitions
├── conf/                        # Configuration files
│   ├── config.yaml              # Main config
│   ├── paths.example.yaml       # Template for local paths
│   ├── attacks/                 # Attack-specific configs
│   ├── datasets/                # Dataset configs
│   └── models/                  # Model configs
├── evaluate/
│   └── visualize_results.ipynb
├── run_attacks.py               # Main attack runner
├── run_judges.py                # Judge evaluation
├── run_sampling.py              # Sampling utilities
└── requirements.txt             # Dependencies
```

## 🔧 Advanced Usage

### Custom Attack Parameters
You can override specific attack parameters:

```bash
python run_attacks.py -m \
    attack=dsm \
    attacks.dsm.num_steps=1000
    attacks.dsm.optimizer=pgm
```

### Custom Generation Parameters
Modify text generation settings - this is useful e.g. for results in https://arxiv.org/abs/2507.04446:

```yaml
generation_config:
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  max_new_tokens: 256
  num_return_sequences: 1
```

## 📈 Results and Analysis

Results are saved under the configured `save_dir` (default `${root_dir}/outputs/${name}/${attack}/results/`).
With `save_format: "default"`, each run is written as:
```
.../YYYY-MM-DD/HHhMMmSSs/{i}/run.json
```
With `save_format: "noDB"` (the default in `conf/config.yaml`), files are named `run-{idx}__{date}__{time}.json` in `save_dir`.

### Visualization & Evaluation (WIP)
Generate plots and analysis with `visualize_results.ipynb` in `evaluate/`

## 🤝 Contributing

Contributions welcome!

## 🙏 Acknowledgments

Please be sure to cite the underlying work if you build on it.

Datasets
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
- [HarmBench](https://github.com/centerforaisafety/HarmBench) for reference attacks & data
- [ORBench](https://arxiv.org/abs/2405.20947)
- [RefusalDirection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)
- [StrongREJECT](https://github.com/dsbowen/strong_reject)
- [XSTest](https://arxiv.org/abs/2308.01263)

Attacks
- [ActorBreaker](https://arxiv.org/abs/2410.10700)
- [AmpleGCG](https://arxiv.org/abs/2404.07921)
- [AutoDAN](https://arxiv.org/abs/2310.04451)
- [BEAST](https://arxiv.org/abs/2402.15570)
- [Best-of-N Jailbreaking](https://arxiv.org/abs/2412.03556)
- [DSM](work in progress, based on https://arxiv.org/abs/2305.11046)
- [GCG](https://arxiv.org/abs/2307.15043)
- [GCG (REINFORCE)](https://arxiv.org/abs/2502.17254)
- [PAIR](https://arxiv.org/abs/2310.08419)
- [PGD (embedding space)](https://arxiv.org/abs/2402.09063)
- [PGD (discrete relaxation)](https://arxiv.org/abs/2402.09154)
- [Human Jailbreaks](https://github.com/centerforaisafety/HarmBench/blob/main/baselines/human_jailbreaks/jailbreaks.py)

Other
- [JudgeZoo](https://github.com/LLM-QC/judgezoo) for judge implementations
