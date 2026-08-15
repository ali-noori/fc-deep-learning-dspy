"""Few-shot (user_message, model fields) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

# (user_message, name, n_classes, in_features after resolution)
# Flat list (not split by execution mode). Built-in CNN vs plugins cnn.py / mlp.py.
CNN_BUILTIN = "CNN"
CNN_PLUGIN = "plugins/models/cnn.py"
MLP_PLUGIN = "plugins/models/mlp.py"

FEWSHOT_PAIRS: list[tuple[str, str, str, str]] = [
    # --- structured / common ---
    (
        "name: CNN\nn_class: 10\nin_features: 1",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        "name: cnn.py\nn_class: 10\nin_features: 1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "name: CNN\nn_classes: 10\nin_features: 1",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        "name: cnn.py\nn_classes: 10\nin_features: 1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "name: mlp.py\nn_classes: 10\nin_features: 784",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "- name: CNN\n- n_classes: 10\n- in_features: 1",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Use the custom plugin model cnn.py (full plugin path).\n"
        "name: plugins/models/cnn.py\nn_classes: 10\nin_features: 1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "name: plugins/models/mlp.py\nn_class: 10\nin_features: 784",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "name: CNN\nn_class: 2\nin_features: 3",
        CNN_BUILTIN,
        "2",
        "3",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        "name: cnn.py\nn_class: 2\nin_features: 3",
        CNN_PLUGIN,
        "2",
        "3",
    ),
    (
        "name: CNN\nn_classes: 100\nin_features: 512",
        CNN_BUILTIN,
        "100",
        "512",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        "name: cnn.py\nn_classes: 100\nin_features: 512",
        CNN_PLUGIN,
        "100",
        "512",
    ),
    (
        "- name: CNN\n- n_class: 4\n- in_features: 8",
        CNN_BUILTIN,
        "4",
        "8",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        "- name: cnn.py\n- n_class: 4\n- in_features: 8",
        CNN_PLUGIN,
        "4",
        "8",
    ),
    (
        'name: "mlp.py"\nn_classes: 5\nin_features: 128',
        MLP_PLUGIN,
        "5",
        "128",
    ),
    # --- natural language / synonyms ---
    (
        "Use built-in CNN architecture with n_classes 10 and in_features 1.",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "I want the custom plugin model cnn.py with 10 classes and 1 input channel for MNIST.",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "Use built-in CNN with 50 output classes and 25 input features.",
        CNN_BUILTIN,
        "50",
        "25",
    ),
    (
        "Use the custom plugin model cnn.py with 50 classes and 25 input features.",
        CNN_PLUGIN,
        "50",
        "25",
    ),
    (
        "I need the built-in CNN model (not the .py plugin) with n_classes 7 and in_features 2.",
        CNN_BUILTIN,
        "7",
        "2",
    ),
    (
        "Use the custom plugin model cnn.py.\n"
        'name: "cnn.py"\nn_classes: 7\nin_features: 2',
        CNN_PLUGIN,
        "7",
        "2",
    ),
    (
        "pls built-in CNN, 10 classes, 1 input feature/channel — not cnn.py",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "custom plugin cnn.py please, 10 classes, in_features=1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "use mlp.py plugin for a fully connected net, 10 classes, 784 in_features",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "architecture CNN (library built-in), output classes 10, input size 1. "
        "I'll choose trainer later.",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "I want plugins/models/cnn.py as the model name, n_class 10, in_features 1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "mlp plugin under plugins/models/mlp.py, classes=10, features=784",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "built-in CNN only. n_classes ten (10), in_features one (1).",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Do NOT use CNN built-in. Use cnn.py plugin. classes 10, in_features 3.",
        CNN_PLUGIN,
        "10",
        "3",
    ),
    (
        "NOT cnn.py — use built-in CNN. classes 8, in_features 1.",
        CNN_BUILTIN,
        "8",
        "1",
    ),
    # --- typos / messy / informal ---
    (
        "name CNNN — wait built-in CNN. n_clas 10, in_featurs 1",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "pls cnn.py plugin (not built-in CNN). n_clas=10 in_featurs=1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "mlp.y — I mean mlp.py. classes 10, in_features 784",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "name:CNN n_class:10 in_features:1",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "name:cnn.py,n_classes:10,in_features:1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "cnn built in, 16 classes, 1 channel",
        CNN_BUILTIN,
        "16",
        "1",
    ),
    (
        "use the .py model cnn.py — 12 classes — 1 in_features",
        CNN_PLUGIN,
        "12",
        "1",
    ),
    (
        "in_features 784, n_classes 10, name mlp.py",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "n_classes 10\nin_features 1\nname CNN",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "n_classes 10\nin_features 1\nname cnn.py",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    # --- edge / boundary / confusing ---
    (
        "name CNN, n_classes 1, in_features 1",
        CNN_BUILTIN,
        "1",
        "1",
    ),
    (
        "cnn.py plugin, n_classes 1, in_features 1",
        CNN_PLUGIN,
        "1",
        "1",
    ),
    (
        "built-in CNN, n_classes 1000, in_features 1",
        CNN_BUILTIN,
        "1000",
        "1",
    ),
    (
        "cnn.py, n_classes 2, in_features 2048",
        CNN_PLUGIN,
        "2",
        "2048",
    ),
    (
        "mlp.py with n_classes 100, in_features 1",
        MLP_PLUGIN,
        "100",
        "1",
    ),
    (
        "I said CNN but I mean the plugin file cnn.py. n_classes 10, in_features 1.",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "Someone wrote cnn.py earlier; ignore that. Use built-in CNN. "
        "n_classes 10, in_features 1.",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Someone wrote MLP; use plugin mlp.py instead. classes 10, in_features 784.",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "name plugins\\models\\cnn.py ; n_class 10 ; in_features 1",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "name plugins\\models\\mlp.py ; n_classes 10 ; in_features 784",
        MLP_PLUGIN,
        "10",
        "784",
    ),
    (
        "Please set name to CNN, n_classes to ten (10), and in_features to one (1).",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "Please set name to cnn.py, n_classes to ten (10), and in_features to one (1).",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "Use CNN for the architecture with 27 classes and 3 input features — "
        "built-in, not plugins/models/cnn.py.",
        CNN_BUILTIN,
        "27",
        "3",
    ),
    (
        "Use plugins/models/cnn.py with 27 classes and 3 input features — "
        "custom plugin, not built-in CNN.",
        CNN_PLUGIN,
        "27",
        "3",
    ),
    (
        "model=mlp.py, classes=3, in_features=64 (plugin MLP, not CNN)",
        MLP_PLUGIN,
        "3",
        "64",
    ),
    (
        "bare name CNN only, 9 classes, 2 in_features (no plugin)",
        CNN_BUILTIN,
        "9",
        "2",
    ),
    (
        "custom cnn.py only, 9 classes, 2 in_features (not built-in CNN)",
        CNN_PLUGIN,
        "9",
        "2",
    ),
    (
        "ok model hyper: CNN built-in; outputs=10; inputs/channels=1. "
        "Also ignore FedAvg and dataset talk.",
        CNN_BUILTIN,
        "10",
        "1",
    ),
    (
        "ok model: cnn.py plugin; outputs=10; inputs=1. ignore trainer choice for now.",
        CNN_PLUGIN,
        "10",
        "1",
    ),
    (
        "mlp.py (plugins/models/mlp.py), n_class=10, in_features=784 for flattened MNIST",
        MLP_PLUGIN,
        "10",
        "784",
    ),
]
