# sample_data_cutomized

Custom MNIST-style splits for federated runs. Shuffle seed: `42`. Keys: `data`, `targets`.

## cross_validation_sample

Source: `sample_data/c1/mnist.npz` only (~30k samples).

1. Split 50/50 → client `c1` / client `c2`
2. Each client half → fold `data/0` / `data/1` (50/50)
3. Each fold → `train.npz` (80%) / `test.npz` (20%)

Use with `logic.mode: "directory"` and `logic.dir: "data"`.

## non_cross_validation_sample

- `c1/`: from `sample_data/c1/mnist.npz` → `train.npz` (80%) / `test.npz` (20%)
- `c2/`: from `sample_data/c2/mnist.npz` → `train.npz` (80%) / `test.npz` (20%)

Use with `logic.mode: "file"` (no fold folders).
