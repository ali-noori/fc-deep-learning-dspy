"""Few-shot (user_message, dataset fields) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

# (user_message, data_dirs, train_dataset_file_name, test_dataset_file_name, logic_dir)
FEWSHOT_PAIRS_FEDERATED: list[tuple[str, str, str, str, str]] = [
    # --- retained / cleaned originals ---
    (
        "data_dir1: sample_data/c1\n"
        "data_dir2: sample_data/c2\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I use two clients: sample_data/c1 and sample_data/c2. "
        "Train file is train.npz, test file is test.npz. "
        "The split folder under mnt/input is data.",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir: sample_data/c1, sample_data/c2\n"
        "train: train.npz\n"
        "test: test.npz\n"
        "logic_dir: data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "- data_dir1: sample_data/c1\n"
        "- data_dir2: sample_data/c2\n"
        "- train_dataset_file_name: train.npy\n"
        "- test_dataset_file_name: test.npy\n"
        "- logic_dir: data",
        "sample_data/c1,sample_data/c2",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "Client folders:\n"
        "  sample_data/c1\n"
        "  sample_data/c2\n"
        "Filenames: train.npz and test.npz inside each fold. logic_dir = data.",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir1: C:/FC/project/data/sample_data/c1\n"
        "data_dir2: C:/FC/project/data/sample_data/c2\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Three-party federated setup:\n"
        "data_dir1: sample_data/c1\n"
        "data_dir2: sample_data/c2\n"
        "data_dir3: sample_data/c3\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data/c1,sample_data/c2,sample_data/c3",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "My MNIST federated data uses logic dir `data`, "
        "train.npz / test.npz per fold, clients at sample_data/c1 and sample_data/c2.",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- new diverse examples ---
    (
        "sample_data/c1, sample_data/c2, train.npz, test.npz, data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "pls use sample_data/c1 n sample_data/c2, trian.npz / test.npz, lgic_dir=data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I also want mlp and FedAvg later, but for now the data layout is: "
        "three hospitals at sample_data/h1, sample_data/h2, sample_data/h3; "
        "each has train.npy and test.npy; the input split folder name is data.",
        "sample_data/h1,sample_data/h2,sample_data/h3",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "client dirs: D:\\UH\\Thesis\\FC\\data\\siteA ; D:\\UH\\Thesis\\FC\\data\\siteB\n"
        "train file train.npz, test file test.npz, logic_dir data",
        "siteA,siteB",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "we got 2 real sites: data/clinic_a and data/clinic_b. "
        "training file called train.csv, eval file test.csv. logic folder data",
        "clinic_a,clinic_b",
        "train.csv",
        "test.csv",
        "data",
    ),
    (
        "dat_dir1=sample_data/c1 dat_dir2=sample_data/c2 "
        "train_dataset_file_name=train.npz test_dataset_file_name=test.npz logic=data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Federated clients live under datasets/part1 and datasets/part2. "
        "Same filenames everywhere: X_train.npz for train and X_test.npz for test. "
        "logic_dir should be data.",
        "datasets/part1,datasets/part2",
        "X_train.npz",
        "X_test.npz",
        "data",
    ),
    (
        "use these folds sample_data/c1 sample_data/c2 sample_data/c3, "
        "files train.npz & test.npz, logic_dir data — ignore model choice for now",
        "sample_data/c1,sample_data/c2,sample_data/c3",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "ok so each partner keeps data local: paths are ./clients/alpha and "
        "./clients/beta. train is train.npz. test is test.npz. logic dir: data",
        "clients/alpha,clients/beta",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data dirs -> sample_data/c1 ; sample_data/c2\n"
        "train -> /mnt/somewhere/train.npz\n"
        "test -> /mnt/somewhere/test.npz\n"
        "logic_dir -> data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "i need federatd data setup with sample_data/c1 + sample_data/c2, "
        "trian.np... wait train.npz and test.npz, logic is da... data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Not simulation. Real multi-client folders at sample_data/hospital1 and "
        "sample_data/hospital2. Training matrix file: train.npz. Held-out: test.npz. "
        "FeatureCloud logic directory name remains data.",
        "sample_data/hospital1,sample_data/hospital2",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- nested project paths + exact extensions ---
    (
        "I have two data directory, first one is "
        "fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data\\c1 "
        "and the second one is "
        "fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data\\c2. "
        "Also the fle named for the test and train are train.npz amd test.npy "
        "and the logic directorz named data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "paths:\n"
        "  ...\\data\\sample_data\\c1\n"
        "  ...\\data\\sample_data\\c2\n"
        "train file: train.npz\n"
        "test file: test.npz\n"
        "logic: data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "first client folder fc-deep-learning-master/data/sample_data/c1, "
        "second fc-deep-learning-master/data/sample_data/c2; "
        "train=train.npz; test=test.npz; logic_dir=data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I have two data dirs under data\\sample_data\\c1 and data\\sample_data\\c2. "
        "IMPORTANT: train is train.npz and test is also test.npz (both npz, not npy). "
        "logic directorz = data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "client1: D:\\UH\\Thesis\\FC\\fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data\\c1 ; "
        "client2: D:\\UH\\Thesis\\FC\\fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data\\c2 ; "
        "train_dataset_file_name train.npz ; test_dataset_file_name test.npz ; logic_dir data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "two folders: data/sample_data/c1 and data/sample_data/c2. "
        "files train.npy and test.npy. logic_dir data",
        "sample_data/c1,sample_data/c2",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "use sample_data/c1 + sample_data/c2. train.npz for training, "
        "but test file is test.npy (misspelled; meant npz). logic=data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "repo-relative: fc-deep-learning-master/fc-deep-learning-master/data/sample_data/c1 "
        "and .../data/sample_data/c2. train.npz / test.npz. logic data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "pls: 1) sample_data\\c1 2) sample_data\\c2. "
        "trian.npz and test.npz. lgic_dir data",
        "sample_data/c1,sample_data/c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I have two data directory first one is data\\sample_data\\hospital1 "
        "second is data\\sample_data\\hospital2. fle names train.npz amd test.npz. "
        "logic directorz named data",
        "sample_data/hospital1,sample_data/hospital2",
        "train.npz",
        "test.npz",
        "data",
    ),
]

# (user_message, data_dir, train_dataset_file_name, test_dataset_file_name, logic_dir)
FEWSHOT_PAIRS_CENTRALIZED: list[tuple[str, str, str, str, str]] = [
    # --- retained / cleaned originals ---
    (
        "data_dir: sample_data_centralized\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I use one directory: sample_data_centralized. "
        "Train file is train.npz, test file is test.npz. "
        "The split folder under mnt/input is data.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "- data_dir: sample_data_centralized\n"
        "- train_dataset_file_name: train.npy\n"
        "- test_dataset_file_name: test.npy\n"
        "- logic_dir: data",
        "sample_data_centralized",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "Single data folder:\n"
        "  sample_data_centralized\n"
        "Filenames: train.npz and test.npz. logic_dir = data.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir1: sample_data_centralized\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "My MNIST centralized data uses logic dir `data`, "
        "train.npz / test.npz, data directory sample_data_centralized.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- new diverse examples ---
    (
        "just sample_data_centralized with train.npz and test.npz, logic data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "one folder only: data/mnist_all, train=train.csv test=test.csv, logic=data",
        "mnist_all",
        "train.csv",
        "test.csv",
        "data",
    ),
    (
        "I dont need clients. Put everything in sample_data_centralized, "
        "trian.npz + test.npz, logic dir data.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "centraliyed run on C:/FC/project/data/sample_data_centralized; "
        "train_dataset_file_name train.npz; test_dataset_file_name test.npz; "
        "logic_dir data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "pls train on single machine. path=datasets/full_set. "
        "files: train.npy, test.npy. logic_dir=data. no federation.",
        "datasets/full_set",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "data_dir sample_data_centralized train_dataset_file_name=train.npz "
        "test_dataset_file_name=test.npz logic_dir=data — also I might pick mlp later",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "all data in one place: ./sample_data_centralized. "
        "training file is /tmp/exports/train.npz and test is /tmp/exports/test.npz. "
        "logic folder name data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "datadir: sample_data_centralized\ntrian: train.npz\ntest: test.npz\nlgic: data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "We are not doing distributed learning. Use sample_data_centralized as the "
        "only data root, with train.npz for training and test.npz for evaluation, "
        "and keep logic_dir as data.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "path D:\\UH\\Thesis\\FC\\data\\sample_data_centralized ; train X_train.npz ; "
        "test X_test.npz ; logic data",
        "sample_data_centralized",
        "X_train.npz",
        "X_test.npz",
        "data",
    ),
    (
        "sample_data_centralized, train.npz, test.npz, data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "ok single-node: folder data/central_run, train file train.pt, "
        "test file test.pt, logic_dir data",
        "central_run",
        "train.pt",
        "test.pt",
        "data",
    ),
    (
        "i want central mode on sample_data_centralized. "
        "train is train.np... train.npz. test.npz. logic da... data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Only one dataset directory called sample_data_centralized. "
        "Inside it the train split file is named train.npz and the test split "
        "is test.npz. FeatureCloud logic dir remains data.",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- nested project paths + exact extensions ---
    (
        "I have one data directory: "
        "fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data_centralized. "
        "The fle named for train and test are train.npz amd test.npz "
        "and the logic directorz named data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "path ...\\data\\sample_data_centralized ; train.npz ; test.npz ; logic data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "single folder fc-deep-learning-master/data/sample_data_centralized, "
        "train=train.npz, test=test.npz, logic_dir=data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "IMPORTANT: both files are npz — train.npz and test.npz (not npy). "
        "data dir sample_data_centralized. logic directorz data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "D:\\UH\\Thesis\\FC\\fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data_centralized "
        "with train_dataset_file_name train.npz and test_dataset_file_name test.npz, logic_dir data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "central folder data/sample_data_centralized, files train.npy and test.npy, logic_dir data",
        "sample_data_centralized",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "sample_data_centralized: train.npz for train, test.npy for test "
        "(test.npy is a typo for test.npz). logic=data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "repo path fc-deep-learning-master/fc-deep-learning-master/data/sample_data_centralized. "
        "train.npz / test.npz. logic data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "pls one dir sample_data_centralized. trian.npz and test.npz. lgic_dir data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I have one data directory data\\sample_data_centralized. "
        "fle names train.npz amd test.npz. logic directorz named data",
        "sample_data_centralized",
        "train.npz",
        "test.npz",
        "data",
    ),
]

# (user_message, data_dir, client_dirs, train_dataset_file_name, test_dataset_file_name, logic_dir)
FEWSHOT_PAIRS_SIMULATION: list[tuple[str, str, str, str, str, str]] = [
    # --- retained / cleaned originals (gold labels corrected where needed) ---
    (
        "data_dir: sample_data_simulation\n"
        "client_dir1: c1\n"
        "client_dir2: c2\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I use one directory: sample_data_simulation. "
        "I use two clients: c1 and c2. "
        "Train file is train.npz, test file is test.npz. "
        "The split folder under mnt/input is data.",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir: sample_data_simulation\n"
        "client_dirs: c1,c2\n"
        "train_dataset_file_name: train.npy\n"
        "test_dataset_file_name: test.npy\n"
        "logic_dir: data",
        "sample_data_simulation",
        "c1,c2",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "- data_dir: sample_data_simulation\n"
        "- cleint_dir1: c1\n"
        "- cleint_dir2: c2\n"
        "- train_dataset_file_name: train.npy\n"
        "- test_dataset_file_name: test.npy\n"
        "- logic_dir: data",
        "sample_data_simulation",
        "c1,c2",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "Data directory: sample_data_simulation\n"
        "Client folders:\n"
        "  c1\n"
        "  c2\n"
        "Filenames: train.npz and test.npz inside each fold. logic_dir = data.",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir: C:/FC/project/data/sample_data_simulation\n"
        "client_dir1: C:/FC/project/data/sample_data_simulation/c1\n"
        "client_dir2: C:/FC/project/data/sample_data_simulation/c2\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Three-party client setup:\n"
        "data_dir: sample_data_simulation\n"
        "client_dir1: c1\n"
        "client_dir2: c2\n"
        "client_dir3: c3\n"
        "train_dataset_file_name: train.npz\n"
        "test_dataset_file_name: test.npz\n"
        "logic_dir: data",
        "sample_data_simulation",
        "c1,c2,c3",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "My MNIST simulation data uses logic dir `data`, "
        "My data directory is sample_data_simulation, "
        "train.npz / test.npz per fold, clients at c1 and c2.",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- new diverse examples ---
    (
        "root sample_data_simulation, clients c1,c2, train.npz test.npz, logic data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I want to run this in simualtion and use sample_data_simulation with "
        "fake clients c1 and c2, same train.npz/test.npz, logic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir=sample_data_simulation; client_dirs=c1,c2,c3; train=train.npy; "
        "test=test.npy; logic_dir=data",
        "sample_data_simulation",
        "c1,c2,c3",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "local fake multi-client: parent folder datasets/sim_root, "
        "subclients alpha and beta, files train.csv / test.csv, logic_dir data",
        "datasets/sim_root",
        "alpha,beta",
        "train.csv",
        "test.csv",
        "data",
    ),
    (
        "pls sim mode. data_dir sample_data_simulation. cleints: c1 c2. "
        "trian.npz test.npz. lgic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Before real federated clients, emulate them under sample_data_simulation "
        "with folders c1, c2, c3. Each fold uses X_train.npz and X_test.npz. "
        "logic_dir is data. Model can wait.",
        "sample_data_simulation",
        "c1,c2,c3",
        "X_train.npz",
        "X_test.npz",
        "data",
    ),
    (
        "sim root: D:\\UH\\Thesis\\FC\\data\\sample_data_simulation\n"
        "clients: client_a, client_b\n"
        "train: train.npz\n"
        "test: test.npz\n"
        "logic: data",
        "sample_data_simulation",
        "client_a,client_b",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "sample_data_simulation + c1/c2 + train.npz + test.npz + data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "dry-run FL locally. parent dir ./sample_data_simulation. "
        "pretend clients named c1 and c2. train file /exports/train.npz, "
        "test file /exports/test.npz. logic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "data_dir sample_data_simulation\n"
        "client_dir1=c1 client_dir2=c2\n"
        "train_dataset_file_name=train.pt\n"
        "test_dataset_file_name=test.pt\n"
        "logic_dir=data",
        "sample_data_simulation",
        "c1,c2",
        "train.pt",
        "test.pt",
        "data",
    ),
    (
        "i need simual... simulation. folder sample_data_simulation. "
        "clients c1 and c2. train.np... train.npz and test.npz. logic da... data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "Not real remote clients — simulate two parties as subfolders c1 and c2 "
        "inside sample_data_simulation. Use train.npz for training splits and "
        "test.npz for testing. Keep logic_dir = data.",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    # --- nested project paths + exact extensions ---
    (
        "I have simulation data directory "
        "fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data_simulation "
        "with clients c1 and c2. The fle named for train and test are train.npz amd test.npz "
        "and the logic directorz named data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "root ...\\data\\sample_data_simulation ; clients c1,c2 ; "
        "train.npz ; test.npz ; logic data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "parent fc-deep-learning-master/data/sample_data_simulation, "
        "clients c1 and c2, train=train.npz, test=test.npz, logic_dir=data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "IMPORTANT: both files are npz — train.npz and test.npz (not npy). "
        "sim root sample_data_simulation, clients c1,c2, logic directorz data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "D:\\UH\\Thesis\\FC\\fc-deep-learning-master\\fc-deep-learning-master\\data\\sample_data_simulation "
        "with fake clients c1,c2 ; train_dataset_file_name train.npz ; "
        "test_dataset_file_name test.npz ; logic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "sim folder data/sample_data_simulation, clients c1 c2, "
        "files train.npy and test.npy, logic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npy",
        "test.npy",
        "data",
    ),
    (
        "sample_data_simulation + c1,c2: train.npz for train, test.npy for test "
        "(test.npy misspelled; should be test.npz). logic=data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "repo path .../data/sample_data_simulation with clients c1 and c2. "
        "train.npz / test.npz. logic data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "pls sim root sample_data_simulation, clients c1 c2. "
        "trian.npz and test.npz. lgic_dir data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
    (
        "I have one sim data directory data\\sample_data_simulation and clients c1,c2. "
        "fle names train.npz amd test.npz. logic directorz named data",
        "sample_data_simulation",
        "c1,c2",
        "train.npz",
        "test.npz",
        "data",
    ),
]
