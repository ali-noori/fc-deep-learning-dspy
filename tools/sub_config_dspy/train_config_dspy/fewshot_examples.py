"""Few-shot (user_message, train_config fields) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

# (user_message, verbose, batch_size, test_batch_size, epochs, lr, batch_count)
# Total few-shots: 50
FEWSHOT_PAIRS: list[tuple[str, str, str, str, str, str, str]] = [
    # 1
    (
        "verbose: false\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 2
    (
        "- verbose: true\n- batch_size: 64\n- test_batch_size: 64\n- epochs: 2\n- lr: 0.01\n- batch_count: 2",
        "true",
        "64",
        "64",
        "2",
        "0.01",
        "2",
    ),
    # 3
    (
        "Keep logging off, train batch 32, test batch 32, one epoch, learning rate 0.1, batch_count 1.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 4
    (
        "verbose true, batch_size 16, test_batch_size 16, epochs 5, lr 0.001, batch_count 4.",
        "true",
        "16",
        "16",
        "5",
        "0.001",
        "4",
    ),
    # 5
    (
        "Quiet training (no verbose). Mini-batch 8 for train and test. Two local epochs. lr=0.05. batch_count=1.",
        "false",
        "8",
        "8",
        "2",
        "0.05",
        "1",
    ),
    # 6
    (
        "I want verbose logging, batch 128, eval batch also 128, 3 epochs, learning rate 0.1, one FedMMb batch.",
        "true",
        "128",
        "128",
        "3",
        "0.1",
        "1",
    ),
    # 7
    (
        "verbose: False\nbatch: 32\ntest batch: 64\nepochs: 1\nlearning rate: 0.1\nbatch_count: 1",
        "false",
        "32",
        "64",
        "1",
        "0.1",
        "1",
    ),
    # 8
    (
        "pls batch_size 32, test_batch_size 32, epochs 1, lr 0.1, batch_count 1, verbose no",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 9
    (
        "Turn verbose on. Use a small laptop batch of 4 for train and test, 1 epoch, lr 0.1, batch_count 1.",
        "true",
        "4",
        "4",
        "1",
        "0.1",
        "1",
    ),
    # 10
    (
        "train_config: verbose=false, batch_size=32, test_batch_size=32, epochs=10, lr=0.01, batch_count=2. Ignore CNN and FedAvg.",
        "false",
        "32",
        "32",
        "10",
        "0.01",
        "2",
    ),
    # 11
    (
        "epochs 2 ; batch_size 64 ; test_batch_size 32 ; lr 0.001 ; batch_count 1 ; verbose false",
        "false",
        "64",
        "32",
        "2",
        "0.001",
        "1",
    ),
    # 12
    (
        "I need print progress (verbose). batch 256 train, 256 test, 4 epochs, step size 0.1, batch_count 8.",
        "true",
        "256",
        "256",
        "4",
        "0.1",
        "8",
    ),
    # 13
    (
        "verbose: true\nbatch_size: 32\ntest_batch_size: 32\nepochs: 2\nlr: 0.1\nbatch_count: 1",
        "true",
        "32",
        "32",
        "2",
        "0.1",
        "1",
    ),
    # 14
    (
        "No logs. Training batch 32, evaluation batch 16, one epoch only, lr=0.1, batch_count=1.",
        "false",
        "32",
        "16",
        "1",
        "0.1",
        "1",
    ),
    # 15
    (
        "batch_count: 2\nepochs: 1\nlr: 0.05\nverbose: false\nbatch_size: 32\ntest_batch_size: 32",
        "false",
        "32",
        "32",
        "1",
        "0.05",
        "2",
    ),
    # 16
    (
        "Use quiet mode, 32 for both batches, 1 epoch, learning rate 0.1, and batch_count 1 for FedMMb.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 17
    (
        "verbose yes, train batch 64, test batch 64, epochs=3, lr=0.0001, batch_count=1",
        "true",
        "64",
        "64",
        "3",
        "0.0001",
        "1",
    ),
    # 18
    (
        "Keep it light: verbose false, batch_size 32, test_batch_size 32, epochs 1, lr 0.1, batch_count 1. Also I'll pick CNN later.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 19
    (
        "verbose: false, bs=16, test_bs=16, ep=2, learning_rate=0.1, batch_count=1",
        "false",
        "16",
        "16",
        "2",
        "0.1",
        "1",
    ),
    # 20
    (
        "I only care about train_config: logging off, 32/32 batches, 1 epoch, lr 0.1, batch_count 1.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 21
    (
        "verbose: true\nbatch_size: 32\ntest_batch_size: 32\nepochs: 2\nlr: 0.1\nbatch_count: 1",
        "true",
        "32",
        "32",
        "2",
        "0.1",
        "1",
    ),
    # 22
    (
        "verbose: false\nbatch_size: 8\ntest_batch_size: 8\nepochs: 1\nlr: 0.01\nbatch_count: 1",
        "false",
        "8",
        "8",
        "1",
        "0.01",
        "1",
    ),
    # 23
    (
        "verbose: true\nbatch_size: 48\ntest_batch_size: 24\nepochs: 6\nlr: 0.05\nbatch_count: 3",
        "true",
        "48",
        "24",
        "6",
        "0.05",
        "3",
    ),
    # 24
    (
        "verbose: false\nbatch_size: 96\ntest_batch_size: 96\nepochs: 4\nlr: 0.001\nbatch_count: 1",
        "false",
        "96",
        "96",
        "4",
        "0.001",
        "1",
    ),
    # 25
    (
        "verbose: true\n"
        "batch_size: 32\n"
        "test_batch_size: 64\n"
        "epochs: 1\n"
        "lr: 0.1\n"
        "batch_count: 1",
        "true",
        "32",
        "64",
        "1",
        "0.1",
        "1",
    ),
    # 26
    (
        "verbose: false\nbatch_size: 512\ntest_batch_size: 512\nepochs: 1\nlr: 0.0005\nbatch_count: 1",
        "false",
        "512",
        "512",
        "1",
        "0.0005",
        "1",
    ),
    # 27
    (
        'verbose: "false"\nbatch_size: "32"\ntest_batch_size: "32"\nepochs: "1"\nlr: "0.1"\nbatch_count: "1"',
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 28
    (
        "verbose: True\nbatch_size: 32\ntest_batch_size: 32\nepochs: 2\nlr: 0.1\nbatch_count: 1",
        "true",
        "32",
        "32",
        "2",
        "0.1",
        "1",
    ),
    # 29
    (
        "train_config:\n"
        "  verbose: false\n"
        "  batch_size: 32\n"
        "  test_batch_size: 32\n"
        "  epochs: 1\n"
        "  lr: 0.1\n"
        "  batch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 30
    (
        "- verbose: false\n- batch_size: 20\n- test_batch_size: 20\n- epochs: 8\n- lr: 0.2\n- batch_count: 1",
        "false",
        "20",
        "20",
        "8",
        "0.2",
        "1",
    ),
    # 31
    (
        "max_iter: 5 and n_classes: 10 and FedAvg — ignore those. "
        "For train_config: verbose: false, batch_size: 32, test_batch_size: 32, "
        "epochs: 1, lr: 0.1, batch_count: 1.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 32
    (
        "Trainer is BasicTrainer with CrossEntropyLoss. train_config is "
        "verbose: true, batch_size: 32, test_batch_size: 32, epochs: 2, lr: 0.1, batch_count: 1.",
        "true",
        "32",
        "32",
        "2",
        "0.1",
        "1",
    ),
    # 33
    (
        "Data is sample_data/c1 with train.npz. train_config: verbose false, "
        "batch_size 32, test_batch_size 32, epochs 1, lr 0.1, batch_count 1.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 34
    (
        "device: gpu can stay default later. verbose: false\nbatch_size: 32\n"
        "test_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 35
    (
        "verbos: false, bacth_size: 32, test_batch_size: 32, epoochs: 1, lrr: 0.1, batch_count: 1 "
        "(meant verbose, batch_size, epochs, lr)",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 36
    (
        "verbose: 0\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 37
    (
        "verbose: 1\nbatch_size: 16\ntest_batch_size: 16\nepochs: 2\nlr: 0.1\nbatch_count: 1",
        "true",
        "16",
        "16",
        "2",
        "0.1",
        "1",
    ),
    # 38
    (
        "verbose: off\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 39
    (
        "verbose: on\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "true",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 40
    (
        "Use the same size 48 for train and test batches. verbose false, 1 epoch, lr 0.1, batch_count 1.",
        "false",
        "48",
        "48",
        "1",
        "0.1",
        "1",
    ),
    # 41
    (
        "FedMMb needs batch_count: 16. Also verbose: false, batch_size: 32, "
        "test_batch_size: 32, epochs: 1, lr: 0.1.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "16",
    ),
    # 42
    (
        "Do not confuse with communication rounds. Local epochs: 3. "
        "verbose: false, batch_size: 32, test_batch_size: 32, lr: 0.1, batch_count: 1.",
        "false",
        "32",
        "32",
        "3",
        "0.1",
        "1",
    ),
    # 43
    (
        '{"verbose": false, "batch_size": 32, "test_batch_size": 32, '
        '"epochs": 1, "lr": 0.1, "batch_count": 1}',
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 44
    (
        "verbose=false batch_size=64 test_batch_size=64 epochs=2 lr=0.01 batch_count=1",
        "false",
        "64",
        "64",
        "2",
        "0.01",
        "1",
    ),
    # 45
    (
        "verbose: false\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 1e-3\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.001",
        "1",
    ),
    # 46
    (
        "Please do not print training logs. eval_batch 8, train batch 32, "
        "epochs 1, learning rate 0.1, batch_count 1.",
        "false",
        "32",
        "8",
        "1",
        "0.1",
        "1",
    ),
    # 47
    (
        "verbose: n\nbatch_size: 32\ntest_batch_size: 32\nepochs: 1\nlr: 0.1\nbatch_count: 1",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
    # 48
    (
        "verbose: y\nbatch_size: 32\ntest_batch_size: 32\nepochs: 2\nlr: 0.1\nbatch_count: 1",
        "true",
        "32",
        "32",
        "2",
        "0.1",
        "1",
    ),
    # 49
    (
        "Copy these names exactly:\n"
        "verbose: false\n"
        "batch_size: 40\n"
        "test_batch_size: 40\n"
        "epochs: 1\n"
        "lr: 0.1\n"
        "batch_count: 2\n"
        "Model name CNN and ImageLoader are not train_config fields.",
        "false",
        "40",
        "40",
        "1",
        "0.1",
        "2",
    ),
    # 50
    (
        "I'd like a laptop-friendly train_config: verbose false, batch_size 32, "
        "test_batch_size 32, one local epoch (not max_iter), lr 0.1, batch_count 1. "
        "Optimizer SGD is already set in trainer.",
        "false",
        "32",
        "32",
        "1",
        "0.1",
        "1",
    ),
]
