"""Few-shot (user_message, hyperparameter fields) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

# (user_message, max_iter, n_classes, federated_model after resolution)
# Balanced: 20 built-in (FedAvg) + 20 custom plugin (FedAvg.py -> resolved path).
FEDAVG_BUILTIN = "FedAvg"
FEDAVG_PLUGIN_PATH = "plugins/aggregators/FedAvg.py"

FEWSHOT_PAIRS: list[tuple[str, str, str, str]] = [
    # =========================
    # Built-in FedAvg (20)
    # =========================
    (
        "max_iter: 10\nn_class: 10\nfederated_model: FedAvg",
        "10",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "- max_iter: 20\n- n_classes: 100\n- federated_model: FedAvg",
        "20",
        "100",
        FEDAVG_BUILTIN,
    ),
    (
        "Use the built-in FedAvg aggregator with 1000 communication rounds and 100 classes.",
        "1000",
        "100",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter: 10000\nn_classes: 50\nfederated_model: FedAvg",
        "10000",
        "50",
        FEDAVG_BUILTIN,
    ),
    (
        "- max_iter: 8\n- n_class: 4\n- federated_model: FedAvg",
        "8",
        "4",
        FEDAVG_BUILTIN,
    ),
    (
        "I need 15 rounds, 7 classes, and the built-in FedAvg aggregator (not the .py plugin).",
        "15",
        "7",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter: 5\nn_classes: 3\nfederated_model: FedAvg",
        "5",
        "3",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter: 500\nn_classes: 25\nfederated_model: FedAvg",
        "500",
        "25",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter: 100\nn_classes: 50\nfederated_model: FedAvg",
        "100",
        "50",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter: 3\nn_classes: 2\nfederated_model: FedAvg",
        "3",
        "2",
        FEDAVG_BUILTIN,
    ),
    (
        "pls 10 rounds, 10 classes, built-in fedavg (not the py file)",
        "10",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "I want max_itr 20, n_clas 10, federated_model FedAvg",
        "20",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "Use FedAvg aggregator, 5 communication rounds, 3 classes. Also I'll pick mlp later.",
        "5",
        "3",
        FEDAVG_BUILTIN,
    ),
    (
        "rounds=12, classes=8, aggregator=FedAvg — built-in library name only",
        "12",
        "8",
        FEDAVG_BUILTIN,
    ),
    (
        "federated_model FedAvg\nn_class 10\nmax_iter 1",
        "1",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "Do NOT use FedAvg.py. Use built-in FedAvg. max_iter 30 and n_classes 10.",
        "30",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "ok so hyper params: communication rounds fifty (50), n_classes=10, fedavg built in",
        "50",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "max_iter 2 ; n_classes 2 ; federated_model: FEDAVG (built-in)",
        "2",
        "2",
        FEDAVG_BUILTIN,
    ),
    (
        "I only care about hyperparams: 200 rounds, 10 classes, FedAvg. "
        "Ignore dataset paths and trainer for now.",
        "200",
        "10",
        FEDAVG_BUILTIN,
    ),
    (
        "max_itr: 7\nn_clas: 9\nfederated_model: FedAvgg — wait, built-in FedAvg",
        "7",
        "9",
        FEDAVG_BUILTIN,
    ),
    # =========================
    # Custom plugin FedAvg.py (20)
    # =========================
    (
        "Use the custom aggregator FedAvg.py.\n"
        "max_iter: 10\nn_classes: 10\nfederated_model: FedAvg.py",
        "10",
        "10",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "- max_iter: 20\n- n_class: 100\n- federated_model: FedAvg.py",
        "20",
        "100",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py with 1000 communication rounds and 100 classes.",
        "1000",
        "100",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "federated_model: FedAvg.py\nmax_iter: 10000\nn_classes: 50",
        "10000",
        "50",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "- max_iter: 8\n- n_class: 4\n- federated_model: FedAvg.py",
        "8",
        "4",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "max_iter: 15\nn_class: 7\nfederated_model: FedAvg.py",
        "15",
        "7",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "max_iter: 5\nn_class: 3\nfederated_model: FedAvg.py",
        "5",
        "3",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py (full plugin path).\n"
        "federated_model: plugins/aggregators/FedAvg.py\nmax_iter: 500\nn_classes: 25",
        "500",
        "25",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        "max_iter: 100\nn_class: 50\nfederated_model: FedAvg.py",
        "100",
        "50",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "Use the custom aggregator FedAvg.py.\n"
        'max_iter: 3\nn_class: 2\nfederated_model: "FedAvg.py"',
        "3",
        "2",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "custom plugin FedAvg.py, 10 rounds, 10 classes",
        "10",
        "10",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "federated_model: plugins/aggregators/FedAvg.py, max_iter 50, n_classes 10",
        "50",
        "10",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "use the custom aggregator file FedAvg.py with 15 rounds and 7 classes",
        "15",
        "7",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "pls use FedAvg.py plugin (not built-in). max_itr=12, n_clas=8",
        "12",
        "8",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "I want the plugin under plugins/aggregators/FedAvg.py. "
        "Communication rounds: 40. Classes: 10.",
        "40",
        "10",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "NOT built-in. Use custom FedAvg.py. rounds 25, n_class 5.",
        "25",
        "5",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "hyper params only: max_iter 9, n_classes 3, federated_model FedAvg.py. "
        "Trainer/model can wait.",
        "9",
        "3",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "aggregator file: FedAvg.py ; iterations/rounds = 60 ; classes = 20",
        "60",
        "20",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "use custom fedavg.py with max_iter 11 and n_classes 10 "
        "(plugin .py, not library FedAvg)",
        "11",
        "10",
        FEDAVG_PLUGIN_PATH,
    ),
    (
        "federated_model='plugins/aggregators/FedAvg.py'\n"
        "max_iter=1\n"
        "n_class=2",
        "1",
        "2",
        FEDAVG_PLUGIN_PATH,
    ),
]
