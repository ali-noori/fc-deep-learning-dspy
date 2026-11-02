# PneumoniaMNIST splits

Same split rules as `sample_data_cutomized/MNIST`. Shuffle seed: `42`. Keys: `data`, `targets`.

Source: official MedMNIST `pneumoniamnist.npz` (5,856 grayscale 28×28 images, 2 classes: 0 = normal, 1 = pneumonia). Train, val, and test were concatenated, shuffled, then split. The original file is under `source/pneumoniamnist.npz`.

Download: https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1

## cross_validation_sample

1. Split 50/50 → client `c1` / client `c2`
2. Each client half → fold `data/0` / `data/1` (50/50)
3. Each fold → `train.npz` (80%) / `test.npz` (20%)

Use with `logic.mode: "directory"` and `logic.dir: "data"`.

Each fold `train.npz` has 1,171 images; each `test.npz` has 293 images.

## non_cross_validation_sample

PneumoniaMNIST is one file, so two client pools are made first (50/50), then:

- `c1/`: `train.npz` (80%) / `test.npz` (20%)
- `c2/`: `train.npz` (80%) / `test.npz` (20%)

Use with `logic.mode: "file"` (no fold folders).

Each client `train.npz` has 2,342 images; each `test.npz` has 586 images.
