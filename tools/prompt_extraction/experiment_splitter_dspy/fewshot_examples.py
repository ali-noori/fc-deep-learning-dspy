"""Few-shot (full request -> five briefs) pairs. Briefs keep the user's wording."""

from __future__ import annotations

# (user_message, execution_mode, dataset, hyper_params, trainer, train_config)
# Model name and new-architecture yes/no are not outputs.
# Total few-shots: 20
FEWSHOT_PAIRS: list[tuple[str, str, str, str, str, str]] = [
    # 1
    (
        "Two hospitals should train on their own machines and only share model weights. Hospital A is sample_data/c1 and hospital B is sample_data/c2. Inside each fold the training set is train.npz and the hold-out set is test.npz. The split folders live in a directory called data. Five communication rounds, ten digit classes, built-in FedAvg. Please use the built-in CNN, grayscale, one input channel. Use BasicTrainer and ImageLoader with CrossEntropyLoss. No verbose logs, train and test batch 32, one local epoch, learning rate 0.1, batch_count 1.",
        "Two hospitals should train on their own machines and only share model weights.",
        "Hospital A is sample_data/c1 and hospital B is sample_data/c2. Inside each fold the training set is train.npz and the hold-out set is test.npz. The split folders live in a directory called data.",
        "Five communication rounds, ten digit classes, built-in FedAvg.",
        "Use BasicTrainer and ImageLoader with CrossEntropyLoss. ten digit classes.",
        "No verbose logs, train and test batch 32, one local epoch, learning rate 0.1, batch_count 1.",
    ),
    # 2
    (
        "Skip federation. Train everything on this one machine as a normal local experiment. Data is under sample_data_centralized, train.npz and test.npz, folds in data. I do not want a new architecture. Use efficientnet_b0.py, 10 classes, 1 channel. max_iter 2 means two epochs per fold here. Built-in FedAvg label can stay. BasicTrainer, ImageLoader, CrossEntropyLoss. Quiet training, batch 32 for train and test, learning rate 0.1, batch_count 1.",
        "Skip federation. Train everything on this one machine as a normal local experiment.",
        "Data is under sample_data_centralized, train.npz and test.npz, folds in data.",
        "max_iter 2. 10 classes. Built-in FedAvg label can stay.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "Quiet training, batch 32 for train and test, learning rate 0.1, batch_count 1. max_iter 2 means two epochs per fold here.",
    ),
    # 3
    (
        "I am not ready for real sites yet. Fake two clients on this laptop so I can debug federated averaging. Use sample_data_simulation, client folders c1 and c2, train.npz, test.npz, folds in data. Please generate a new small CNN for 28x28 images. 10 rounds, 10 classes, FedAvg not the py plugin. BasicTrainer, ImageLoader, CrossEntropyLoss. verbose true, batch_size 16, test batch 16, 1 epoch, lr 0.01, batch_count 1.",
        "I am not ready for real sites yet. Fake two clients on this laptop so I can debug federated averaging.",
        "Use sample_data_simulation, client folders c1 and c2, train.npz, test.npz, folds in data.",
        "10 rounds, 10 classes, FedAvg not the py plugin.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "verbose true, batch_size 16, test batch 16, 1 epoch, lr 0.01, batch_count 1.",
    ),
    # 4
    (
        "Batch size 8. Real federated clients, not a simulation. Then the test batch is 8 as well, one local epoch, lr 0.1, batch_count 2, no logs. Sites are sample_data/c1 and sample_data/c2, files train.npz and test.npz, directory data. 3 communication rounds, 10 classes, built-in FedAvg. Existing cnn.py, one channel. Do not write a new model. BasicTrainer, ImageLoader, CrossEntropyLoss.",
        "Real federated clients, not a simulation.",
        "Sites are sample_data/c1 and sample_data/c2, files train.npz and test.npz, directory data.",
        "3 communication rounds, 10 classes, built-in FedAvg.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "Batch size 8. Then the test batch is 8 as well, one local epoch, lr 0.1, batch_count 2, no logs.",
    ),
    # 5
    (
        "Centralized run on sample_data_cutomized/cross_validation_sample/c1 only. train.npz and test.npz under fold folders data/0 and data/1. Use mobilenetv3_small.py, 10 classes, in_features 1. No new model file. Two epochs per fold. FedAvg. BasicTrainer and ImageLoader, loss CrossEntropyLoss. verbose false, both batches 32, learning rate 0.1, batch_count 1.",
        "Centralized run.",
        "sample_data_cutomized/cross_validation_sample/c1 only. train.npz and test.npz under fold folders data/0 and data/1.",
        "Two epochs per fold. 10 classes. FedAvg.",
        "BasicTrainer and ImageLoader, loss CrossEntropyLoss. 10 classes.",
        "verbose false, both batches 32, learning rate 0.1, batch_count 1.",
    ),
    # 6
    (
        "Simulate clients locally before calling any hospital. Folder sample_data_simulation with c1 and c2. train.npy and test.npy. logic folder data. I want you to design a new MLP, not pick cnn.py. 15 rounds, 2 classes, custom aggregator FedAvg.py. Trainer plugin FedMMb.py, RnaLoader.py, focal_loss.py. Print progress, batch 64 train, batch 64 test, epochs 2, lr 0.001, batch_count 4.",
        "Simulate clients locally before calling any hospital.",
        "Folder sample_data_simulation with c1 and c2. train.npy and test.npy. logic folder data.",
        "15 rounds, 2 classes, custom aggregator FedAvg.py.",
        "Trainer plugin FedMMb.py, RnaLoader.py, focal_loss.py. 2 classes.",
        "Print progress, batch 64 train, batch 64 test, epochs 2, lr 0.001, batch_count 4.",
    ),
    # 7
    (
        "Federated learning across sample_data/c1, sample_data/c2, and sample_data/c3. Each split uses train.npz and test.npz inside data. Keep the built-in CNN. 100 classes, 3 input channels. 1000 communication rounds, built-in FedAvg. BasicTrainer, ImageLoader, CrossEntropyLoss. verbose on, batch_size 128, test_batch_size 128, epochs 1, lr 0.1, batch_count 1.",
        "Federated learning across sites.",
        "sample_data/c1, sample_data/c2, and sample_data/c3. Each split uses train.npz and test.npz inside data.",
        "1000 communication rounds, 100 classes, built-in FedAvg.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 100 classes.",
        "verbose on, batch_size 128, test_batch_size 128, epochs 1, lr 0.1, batch_count 1.",
    ),
    # 8
    (
        "One PC, no fake clients, no remote hospitals: centralized. sample_data_centralized, train.npz, test.npz, dir data. resnet18.py already exists, 10 classes, 1 channel. max_iter 2, FedAvg. BasicTrainer, ImageLoader, CrossEntropyLoss. no logs, batch 32, test batch 32, lr 0.1, batch_count 1.",
        "One PC, no fake clients, no remote hospitals: centralized.",
        "sample_data_centralized, train.npz, test.npz, dir data.",
        "max_iter 2, 10 classes, FedAvg.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "no logs, batch 32, test batch 32, lr 0.1, batch_count 1.",
    ),
    # 9
    (
        "Dry-run federated averaging on one laptop. sample_data_simulation, clients c1 and c2, train.npz, test.npz, folds called data. Do not generate a new network. Use CNN, 10 classes, in_features 1. 5 rounds, FedAvg. BasicTrainer, ImageLoader, CrossEntropyLoss. verbose false, batch 4 because the laptop is small, test batch 4, 1 epoch, learning rate 0.1, batch_count 1.",
        "Dry-run federated averaging on one laptop.",
        "sample_data_simulation, clients c1 and c2, train.npz, test.npz, folds called data.",
        "5 rounds, 10 classes, FedAvg.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "verbose false, batch 4 because the laptop is small, test batch 4, 1 epoch, learning rate 0.1, batch_count 1.",
    ),
    # 10
    (
        "Please write a new convolutional architecture, then we will train it federated. Clients sample_data/c1 and sample_data/c2. train.npz, test.npz, split dir data. 8 rounds, 10 classes, built-in FedAvg. The new net should be for grayscale digits. BasicTrainer, ImageLoader, CrossEntropyLoss. Logging off. batch_size 32, test_batch_size 16, epochs 1, lr 0.05, batch_count 1.",
        "train it federated.",
        "Clients sample_data/c1 and sample_data/c2. train.npz, test.npz, split dir data.",
        "8 rounds, 10 classes, built-in FedAvg.",
        "BasicTrainer, ImageLoader, CrossEntropyLoss. 10 classes.",
        "Logging off. batch_size 32, test_batch_size 16, epochs 1, lr 0.05, batch_count 1.",
    ),
    # 11
    (
        "Centralized fine-tune. Data root sample_data_centralized. Files are train.npz and test.npz. Fold directory is data. Use the pretrained EfficientNet-B0 plugin, 10 classes, one channel. Not a newly written class. max_iter 2, aggregator FedAvg. Built-in BasicTrainer, built-in ImageLoader, CrossEntropyLoss. verbose: false, batch_size: 32, test_batch_size: 32, epochs follow max_iter, lr: 0.1, batch_count: 1.",
        "Centralized fine-tune.",
        "Data root sample_data_centralized. Files are train.npz and test.npz. Fold directory is data.",
        "max_iter 2, 10 classes, aggregator FedAvg.",
        "Built-in BasicTrainer, built-in ImageLoader, CrossEntropyLoss. 10 classes.",
        "verbose: false, batch_size: 32, test_batch_size: 32, epochs follow max_iter, lr: 0.1, batch_count: 1.",
    ),
    # 12
    (
        "lr 0.001 and batch_count 8, but that is only the local trainer setup. The collaboration itself is federated: two real clients sample_data/c1 and sample_data/c2. train.npz, test.npz, folder data. 20 communication rounds, 7 classes, FedAvg.py plugin. Generate a new model. FedMMb.py trainer, ImageLoader.py loader, NLLLoss. verbose yes, train batch 32, eval batch 32, local epochs 3.",
        "The collaboration itself is federated: two real clients.",
        "two real clients sample_data/c1 and sample_data/c2. train.npz, test.npz, folder data.",
        "20 communication rounds, 7 classes, FedAvg.py plugin.",
        "FedMMb.py trainer, ImageLoader.py loader, NLLLoss. 7 classes.",
        "lr 0.001 and batch_count 8. verbose yes, train batch 32, eval batch 32, local epochs 3.",
    ),
    # 13
    (
        "- Run federated learning on real hospital clients.\n- data_dir1: sample_data/c1\n- data_dir2: sample_data/c2\n- train_dataset_file_name: train.npz\n- test_dataset_file_name: test.npz\n- logic_dir: data\n- max_iter: 10\n- n_classes: 2\n- federated_model: FedAvg\n- Do not generate a new architecture.\n- name: tabnet.py\n- in_features: 784\n- trainer: BasicTrainer\n- data_loader: RnaLoader.py\n- loss.name: BCEWithLogitsLoss\n- verbose: false\n- batch_size: 64\n- test_batch_size: 64\n- epochs: 1\n- lr: 0.001\n- batch_count: 1",
        "Run federated learning on real hospital clients.",
        "data_dir1: sample_data/c1. data_dir2: sample_data/c2. train_dataset_file_name: train.npz. test_dataset_file_name: test.npz. logic_dir: data.",
        "max_iter: 10. n_classes: 2. federated_model: FedAvg.",
        "trainer: BasicTrainer. data_loader: RnaLoader.py. loss.name: BCEWithLogitsLoss. n_classes: 2.",
        "verbose: false. batch_size: 64. test_batch_size: 64. epochs: 1. lr: 0.001. batch_count: 1.",
    ),
    # 14
    (
        "{\"mode\": \"centralized\", \"data_dir\": \"sample_data_centralized\", \"train\": \"train.npz\", \"test\": \"test.npz\", \"logic_dir\": \"data\", \"max_iter\": 5, \"n_class\": 10, \"federated_model\": \"FedAvg\", \"model\": \"ft_transformer.py\", \"in_features\": 128, \"new_architecture\": false, \"trainer\": \"BasicTrainer\", \"data_loader\": \"ImageLoader\", \"loss\": \"MSELoss\", \"verbose\": \"n\", \"batch_size\": 32, \"test_batch_size\": 16, \"epochs\": 2, \"lr\": 0.01, \"batch_count\": 1}",
        "centralized",
        "data_dir: sample_data_centralized. train: train.npz. test: test.npz. logic_dir: data.",
        "max_iter: 5. n_class: 10. federated_model: FedAvg.",
        "trainer: BasicTrainer. data_loader: ImageLoader. loss: MSELoss. n_class: 10.",
        "verbose: n. batch_size: 32. test_batch_size: 16. epochs: 2. lr: 0.01. batch_count: 1.",
    ),
    # 15
    (
        "I already saved customized_model_v2.py. Do not write a new model. Simulate four fake clients c1, c2, c3, and c4 under sample_data_simulation. Training file is train.npz but the test file is test.npy. Fold folder name is data. 25 rounds, n_class 10, use the plugin aggregator FedAvg.py. The saved model has 10 classes and in_features 1. Trainer plugin FedMMb.py, built-in ImageLoader without .py, CrossEntropyLoss. verbose 0, train batch 32, test batch 32, 1 local epoch, lr 0.1, batch_count 2.",
        "Simulate four fake clients.",
        "four fake clients c1, c2, c3, and c4 under sample_data_simulation. Training file is train.npz but the test file is test.npy. Fold folder name is data.",
        "25 rounds, n_class 10, use the plugin aggregator FedAvg.py.",
        "Trainer plugin FedMMb.py, built-in ImageLoader without .py, CrossEntropyLoss. n_class 10.",
        "verbose 0, train batch 32, test batch 32, 1 local epoch, lr 0.1, batch_count 2.",
    ),
    # 16
    (
        "batch_size 8 and test_batch_size 32, epochs 1, lr 0.05, batch_count 1, verbose off. Loss is NLLLoss. Loader ImageLoader. Trainer BasicTrainer. Built-in CNN, not the cnn.py plugin, 10 classes, 1 channel. Please do not generate a new network. max_iter 3, n_classes 10, built-in FedAvg. The hospital folder is sample_data_cutomized/non_cross_validation_sample/c1. Files are train.npz and test.npz. logic_dir data. This is centrlaized, one machine, no fake clients.",
        "This is centrlaized, one machine, no fake clients.",
        "The hospital folder is sample_data_cutomized/non_cross_validation_sample/c1. Files are train.npz and test.npz. logic_dir data.",
        "max_iter 3, n_classes 10, built-in FedAvg.",
        "Loss is NLLLoss. Loader ImageLoader. Trainer BasicTrainer. 10 classes.",
        "batch_size 8 and test_batch_size 32, epochs 1, lr 0.05, batch_count 1, verbose off.",
    ),
    # 17
    (
        "Actual distributed clients connected through the controller, not a simulation. Sites sample_data/c1 and sample_data/c2. train.npz, test.npz, logic_dir data. 50 communication rounds. n_classes 10. custom aggregator FedAvg.py. Use plugins/models/pre-trained_models/cnn_architectures/resnet18.py, 3 input channels. No newly written class. data_loader ImageLoader.py. BasicTrainer. CrossEntropyLoss. verbose: 1. batch_size 16. test_batch_size 16. epochs 1. lr 0.1. batch_count 1.",
        "Actual distributed clients connected through the controller, not a simulation.",
        "Sites sample_data/c1 and sample_data/c2. train.npz, test.npz, logic_dir data.",
        "50 communication rounds. n_classes 10. custom aggregator FedAvg.py.",
        "data_loader ImageLoader.py. BasicTrainer. CrossEntropyLoss. n_classes 10.",
        "verbose: 1. batch_size 16. test_batch_size 16. epochs 1. lr 0.1. batch_count 1.",
    ),
    # 18
    (
        "One dataset on one machine only: sample_data_centralized, train.npy, test.npy, logic_dir data. tabtransformer.py already exists, 4 classes, in_features 32. I do not want a new architecture. max_iter 2. FedAvg. RnaLoader. BasicTrainer. BCEWithLogitsLoss. verbose n. both batches 8. learning rate 0.0001. batch_count 1. local epochs 2.",
        "One dataset on one machine only.",
        "sample_data_centralized, train.npy, test.npy, logic_dir data.",
        "max_iter 2. 4 classes. FedAvg.",
        "RnaLoader. BasicTrainer. BCEWithLogitsLoss. 4 classes.",
        "verbose n. both batches 8. learning rate 0.0001. batch_count 1. local epochs 2.",
    ),
    # 19
    (
        "lr 0.0005, batch_count 8, verbose yes, train batch 128, eval batch 64, local epochs 4. In the middle: please design a new MLP, in_features 784, 5 classes. Federated across sample_data/c1 and sample_data/c2. train.npz and test.npz. directory called data. 100 rounds. built-in FedAvg. FedMMb.py trainer with built-in ImageLoader and CrossEntropyLoss.",
        "Federated across sample_data/c1 and sample_data/c2.",
        "sample_data/c1 and sample_data/c2. train.npz and test.npz. directory called data.",
        "100 rounds. 5 classes. built-in FedAvg.",
        "FedMMb.py trainer with built-in ImageLoader and CrossEntropyLoss. 5 classes.",
        "lr 0.0005, batch_count 8, verbose yes, train batch 128, eval batch 64, local epochs 4.",
    ),
    # 20
    (
        "pls a dry-run on one laptop with fake clients c1 and c2 in sample_data_simulation; train.npz; test.npz; logic_dir data. mobilenetv3_small.py, 10 classes, in_features 1 — not a new file. max_iter 10000 for the rounds, but local epochs stay 1. n_classes 10. FedAvg not FedAvg.py. BasicTrainer, ImageLoader, focal_loss.py. verbose no. batch 32 train. batch 64 test. lr 0.1. batch_count 1.",
        "a dry-run on one laptop with fake clients.",
        "fake clients c1 and c2 in sample_data_simulation; train.npz; test.npz; logic_dir data.",
        "max_iter 10000 for the rounds. n_classes 10. FedAvg not FedAvg.py.",
        "BasicTrainer, ImageLoader, focal_loss.py. n_classes 10.",
        "local epochs stay 1. verbose no. batch 32 train. batch 64 test. lr 0.1. batch_count 1.",
    ),
]
