"""Few-shot (user_message, trainer fields) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

# (user_message, name, data_loader, loss_name, num_classes after resolution)
# Flat list of 60 diverse examples (not split by execution mode).
TRAINER_BUILTIN = "BasicTrainer"
TRAINER_PLUGIN = "plugins/trainers/FedMMb.py"
DATALOADER_BUILTIN = "ImageLoader"
DATALOADER_PLUGIN = "plugins/dataloaders/ImageLoader.py"
DATALOADER_PLUGIN_ALT = "plugins/dataloaders/RnaLoader.py"
LOSS_BUILTIN = "CrossEntropyLoss"
LOSS_BUILTIN_ALT = "NLLLoss"
LOSS_BUILTIN_ALT2 = "BCEWithLogitsLoss"
LOSS_BUILTIN_ALT3 = "MSELoss"
LOSS_PLUGIN = "plugins/loss/focal_loss.py"

FEWSHOT_PAIRS: list[tuple[str, str, str, str, str]] = [
    # --- all built-in ---
    (
        "name: BasicTrainer\ndata_loader: ImageLoader\nloss.name: CrossEntropyLoss\nn_class: 10",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "Use built-in BasicTrainer (not the .py plugin) with ImageLoader, "
        "CrossEntropyLoss, and 3 classes.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "3",
    ),
    (
        "- name: BasicTrainer\n- data_loader: ImageLoader\n"
        "- loss.name: CrossEntropyLoss\n- n_class: 50",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "50",
    ),
    (
        "BasicTrainer, ImageLoader, built-in BCEWithLogitsLoss loss, 2 classes.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN_ALT2,
        "2",
    ),
    (
        "- name: BasicTrainer\n- data_loader: ImageLoader\n"
        "- loss.name: MSELoss\n- n_class: 50",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN_ALT3,
        "50",
    ),
    (
        "name: BasicTrainer\ndata_loader: ImageLoader\nloss.name: NLLLoss\nn_classes: 7",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN_ALT,
        "7",
    ),
    (
        "I want built-in CrossEntropyLoss (not focal_loss.py) with BasicTrainer, "
        "ImageLoader, n_class 3.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "3",
    ),
    (
        "pls all built-in: BasicTrainer + ImageLoader + CrossEntropyLoss, 10 classes "
        "(no .py plugins)",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "n_classes 1\nloss.name CrossEntropyLoss\ndata_loader ImageLoader\nname BasicTrainer",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "1",
    ),
    # --- all plugin ---
    (
        "Use custom plugins: FedMMb.py trainer, ImageLoader.py dataloader, "
        "focal_loss.py loss, n_class 10.",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "name: FedMMb.py\ndata_loader: ImageLoader.py\n"
        "loss.name: focal_loss.py\nn_classes: 10",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "full paths please:\n"
        "name: plugins/trainers/FedMMb.py\n"
        "data_loader: plugins/dataloaders/ImageLoader.py\n"
        "loss.name: plugins/loss/focal_loss.py\n"
        "n_class: 5",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "5",
    ),
    (
        "all custom: FedMMb.py + RnaLoader.py + focal_loss.py, classes 8",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_PLUGIN,
        "8",
    ),
    (
        "plugin stack FedMMb.py / ImageLoader.py / focal_loss.py — 100 classes",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "100",
    ),
    (
        'name: "FedMMb.py"\ndata_loader: "ImageLoader.py"\n'
        'loss.name: "focal_loss.py"\nn_class: 2',
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "2",
    ),
    # --- trainer plugin focus (dataloader/loss built-in) ---
    (
        "Use the custom plugin trainer FedMMb.py.\n"
        "name: FedMMb.py\ndata_loader: ImageLoader\nloss.name: CrossEntropyLoss\nn_class: 10",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "Use the custom plugin trainer FedMMb.py with ImageLoader, "
        "CrossEntropyLoss, and 3 classes.",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "3",
    ),
    (
        "Use the custom plugin trainer FedMMb.py (full plugin path).\n"
        "name: plugins/trainers/FedMMb.py\ndata_loader: ImageLoader\n"
        "loss.name: CrossEntropyLoss\nn_class: 50",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "50",
    ),
    (
        "NOT BasicTrainer — use FedMMb.py plugin. ImageLoader + CrossEntropyLoss, n_class 7.",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "7",
    ),
    (
        "Someone wrote BasicTrainer; ignore that. Use FedMMb.py. "
        "ImageLoader, CrossEntropyLoss, classes 12.",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "12",
    ),
    (
        "built-in BasicTrainer only (not FedMMb.py), ImageLoader, CrossEntropyLoss, 9 classes",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "9",
    ),
    # --- dataloader plugin focus ---
    (
        "Use the custom plugin dataloader ImageLoader.py.\n"
        "name: BasicTrainer\ndata_loader: ImageLoader.py\n"
        "loss.name: CrossEntropyLoss\nn_class: 10",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "BasicTrainer with custom plugin dataloader RnaLoader.py, "
        "CrossEntropyLoss, 100 classes.",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_BUILTIN,
        "100",
    ),
    (
        "Use the custom plugin dataloader ImageLoader.py (full plugin path).\n"
        "name: BasicTrainer\ndata_loader: plugins/dataloaders/ImageLoader.py\n"
        "loss.name: CrossEntropyLoss\nn_classes: 8",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN,
        LOSS_BUILTIN,
        "8",
    ),
    (
        "Use the custom plugin dataloader RnaLoader.py.\n"
        'name: BasicTrainer\ndata_loader: "RnaLoader.py"\n'
        "loss.name: CrossEntropyLoss\nn_class: 25",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_BUILTIN,
        "25",
    ),
    (
        "Do NOT use ImageLoader.py. Use built-in ImageLoader. "
        "BasicTrainer, CrossEntropyLoss, n_class 6.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "6",
    ),
    (
        "I said ImageLoader but I mean the plugin ImageLoader.py. "
        "BasicTrainer, CrossEntropyLoss, classes 10.",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "dataloader plugins\\dataloaders\\RnaLoader.py with BasicTrainer and "
        "CrossEntropyLoss, n_classes 15",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_BUILTIN,
        "15",
    ),
    # --- loss plugin focus ---
    (
        "Use the custom plugin loss focal_loss.py.\n"
        "name: BasicTrainer\ndata_loader: ImageLoader\n"
        "loss.name: focal_loss.py\nn_class: 10",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "BasicTrainer, ImageLoader, custom plugin loss focal_loss.py, 2 classes.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "2",
    ),
    (
        "Use the custom plugin loss focal_loss.py (full plugin path).\n"
        "name: BasicTrainer\ndata_loader: ImageLoader\n"
        "loss.name: plugins/loss/focal_loss.py\nn_class: 50",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "50",
    ),
    (
        "Use the custom plugin loss focal_loss.py.\n"
        "name: BasicTrainer\ndata_loader: ImageLoader\n"
        'loss.name: "focal_loss.py"\nn_class: 3',
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "3",
    ),
    (
        "built-in CrossEntropyLoss only — not focal_loss.py. "
        "BasicTrainer, ImageLoader, classes 11.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "11",
    ),
    (
        "I said CrossEntropyLoss earlier; actually use focal_loss.py plugin. "
        "BasicTrainer + ImageLoader, n_class 10.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "10",
    ),
    # --- mixed combinations ---
    (
        "FedMMb.py trainer, built-in ImageLoader, CrossEntropyLoss, n_class 10",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "BasicTrainer, ImageLoader.py plugin, CrossEntropyLoss, n_class 10",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "BasicTrainer, ImageLoader, focal_loss.py, n_class 10",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "FedMMb.py + ImageLoader.py + CrossEntropyLoss, classes 10 "
        "(trainer+loader plugins, built-in loss)",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "FedMMb.py + ImageLoader + focal_loss.py, classes 10 "
        "(trainer+loss plugins, built-in loader)",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "BasicTrainer + ImageLoader.py + focal_loss.py, classes 10 "
        "(loader+loss plugins, built-in trainer)",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "FedMMb.py with RnaLoader.py and NLLLoss, n_classes 20",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_BUILTIN_ALT,
        "20",
    ),
    (
        "BasicTrainer, RnaLoader.py, MSELoss, n_class 1",
        TRAINER_BUILTIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_BUILTIN_ALT3,
        "1",
    ),
    (
        "plugin trainer FedMMb.py, built-in ImageLoader, BCEWithLogitsLoss, 2 classes",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN_ALT2,
        "2",
    ),
    (
        "mixed: name plugins/trainers/FedMMb.py, data_loader ImageLoader, "
        "loss.name plugins/loss/focal_loss.py, n_classes 16",
        TRAINER_PLUGIN,
        DATALOADER_BUILTIN,
        LOSS_PLUGIN,
        "16",
    ),
    # --- messy / typos / ambiguous / edge ---
    (
        "pls BasicTriner — wait BasicTrainer; ImageLoader; CrossEntropyLoss; 10 classes",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "fedmmb.py plugin trainer, imageloader.py plugin loader, "
        "focal_loss.py, n_clas 10",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "name:BasicTrainer data_loader:ImageLoader loss.name:CrossEntropyLoss n_class:10",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "name:FedMMb.py,data_loader:ImageLoader.py,loss.name:focal_loss.py,n_classes:10",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "classes 10, loss CrossEntropyLoss, loader ImageLoader, trainer BasicTrainer",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "classes 10, loss focal_loss.py, loader ImageLoader.py, trainer FedMMb.py",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "Bare ImageLoader (built-in, no .py), BasicTrainer, CrossEntropyLoss, 999 classes",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "999",
    ),
    (
        "Bare names only: BasicTrainer / ImageLoader / CrossEntropyLoss / 1 class",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "1",
    ),
    (
        "Ignore dataset and FedAvg talk. Trainer=BasicTrainer, loader=ImageLoader, "
        "loss=CrossEntropyLoss, n_classes=10.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "Ignore model mlp. Use FedMMb.py, ImageLoader.py, focal_loss.py, n_class=10.",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "loss_name: CrossEntropyLoss\ntrainer: BasicTrainer\n"
        "data_loader: ImageLoader\nn_class: 14",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "14",
    ),
    (
        "loss_name: focal_loss.py\ntrainer: FedMMb.py\n"
        "data_loader: RnaLoader.py\nn_class: 14",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN_ALT,
        LOSS_PLUGIN,
        "14",
    ),
    (
        "Please set trainer to BasicTrainer, dataloader to ImageLoader, "
        "loss to CrossEntropyLoss, and classes to ten (10).",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "10",
    ),
    (
        "Please set trainer to FedMMb.py, dataloader to ImageLoader.py, "
        "loss to focal_loss.py, and classes to ten (10).",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "10",
    ),
    (
        "Use BasicTrainer with ImageLoader and CrossEntropyLoss — built-in stack, "
        "not plugins/trainers/FedMMb.py / ImageLoader.py / focal_loss.py. n_class 18.",
        TRAINER_BUILTIN,
        DATALOADER_BUILTIN,
        LOSS_BUILTIN,
        "18",
    ),
    (
        "Use plugins/trainers/FedMMb.py with plugins/dataloaders/ImageLoader.py and "
        "plugins/loss/focal_loss.py — full plugin stack, not built-ins. n_class 18.",
        TRAINER_PLUGIN,
        DATALOADER_PLUGIN,
        LOSS_PLUGIN,
        "18",
    ),
]
