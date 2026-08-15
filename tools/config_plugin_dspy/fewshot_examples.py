"""Few-shot (user_request, intent_dict) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

FEWSHOT_PAIRS: list[tuple[str, dict[str, str]]] = [
    ("Use the CNN model with the FedMMb trainer", {"model_name": "cnn.py", "trainer_name": "FedMMb.py"}),
    ("Switch to cnn.py and use FedMMb", {"model_name": "cnn.py", "trainer_name": "FedMMb.py"}),
    ("Switch to cnn.py", {"model_name": "cnn.py"}),
    ("Please switch to CNN", {"model_name": "cnn.py"}),
    ("Switch to the MLP model", {"model_name": "mlp.py"}),
    ("Prefer cnn.py and keep current trainer", {"model_name": "cnn.py"}),
    ("Set the trainer to FedMMb for training", {"trainer_name": "FedMMb.py"}),
    ("Use mlp.py with FedMMb", {"model_name": "mlp.py", "trainer_name": "FedMMb.py"}),
    ("Use built-in FedAvg for aggregation", {"aggregator_name": "FedAvg"}),
    ("Use the FedAvg plugin from aggregators", {"aggregator_name": "FedAvg.py"}),
    (
        "Switch model to mlp.py and use FedAvg averaging",
        {"model_name": "mlp.py", "aggregator_name": "FedAvg"},
    ),
    ("Use cross entropy loss", {"loss_name": "CrossEntropyLoss"}),
    ("Switch to focal loss", {"loss_name": "focal_loss.py"}),
    ("Use focal_loss.py for training", {"loss_name": "focal_loss.py"}),
    ("Use the image loader", {"data_loader_name": "ImageLoader.py"}),
    ("Switch to RnaLoader for tabular data", {"data_loader_name": "RnaLoader.py"}),
    ("Use built-in ImageLoader without .py suffix", {"data_loader_name": "ImageLoader"}),
    (
        "Use model mlp.py, trainer FedMMb.py, federated aggregator FedAvg, "
        "loss focal_loss.py, and data loader ImageLoader.py",
        {
            "model_name": "mlp.py",
            "trainer_name": "FedMMb.py",
            "aggregator_name": "FedAvg",
            "loss_name": "focal_loss.py",
            "data_loader_name": "ImageLoader.py",
        },
    ),
]
