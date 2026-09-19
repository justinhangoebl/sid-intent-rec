# sid-intent-rec

Intent-aware recommender systems with semantic IDs (master thesis).

Two stages, both driven by a YAML config:

1. **quantizer**: turn item features into semantic IDs (e.g. residual k-means)
2. **recommender**: train and evaluate a recommender on user-item interactions

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/).

```
uv venv --prompt intent-rec --version PYTHON_VERSION
uv sync
```

## Usage

```
uv run quantizer   {train,infer,optimize} --quantizer RKMeans --config configs/sid.yaml
uv run recommender {train,infer,optimize} --model TopPop --config configs/rec.yaml
```

The task and model/quantizer arguments are parsed, but nothing is dispatched to them yet.

## Layout

```
configs/     YAML configs
dataset/     ml1m, onion, onion-100k
pipeline/
  cli.py       entry points (recommender, quantizer)
  config.py    defaults and config loading
  models/      recommenders
  quantizers/  semantic-ID quantizers
  trainers/    training loop
  metrics/     ranking metrics and evaluator
  utils/       logging, seeding, wandb
```

## Config

Configs are plain YAML. Missing keys fall back to the defaults in `pipeline/config.py`. Runs are logged to
Weights & Biases when `run.wandb: true`; set it to `false` to disable.
