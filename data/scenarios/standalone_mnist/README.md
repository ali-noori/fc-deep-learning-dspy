# Standalone MNIST (centralized)

Self-contained FeatureCloud-style input for running the deep learning app
**without** the FeatureCloud controller (`STANDALONE=1`).

## Layout

- `input/` — mounted to `/mnt/input` (`config.yml`, data, plugins the config references)
- `output/` — mounted to `/mnt/output` (predictions, model, copied config)

Training reads from `input/` and writes only under `output/`.

## Contents

| File | Role |
|------|------|
| `config.yml` | `centralized: True` (also forced by standalone) |
| `mnist.npz` | Train/test images |
| `ImageLoader.py` | DataLoader plugin |
| `cnn.py` | Model plugin |
| `FedMMb.py` | Local trainer plugin |
| `FedAvg.py` | Aggregator plugin (unused in centralized, but named in config) |
