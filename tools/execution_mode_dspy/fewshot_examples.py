"""Few-shot (user_message, execution_mode) pairs for BootstrapFewShot (data only)."""

from __future__ import annotations

FEWSHOT_PAIRS: list[tuple[str, str]] = [
    # Federated (20)
    ("I want to run the federated mode.", "federated"),
    ("Please use federated learning with two clients.", "federated"),
    ("Train across multiple hospitals / clients.", "federated"),
    ("I need distributed learning with real clients.", "federated"),
    ("fed", "federated"),
    ("federated", "federated"),
    ("fedarated", "federated"),
    ("go with federation", "federated"),
    ("multi client training please", "federated"),
    ("we have 3 real sites and each keeps its own data", "federated"),
    (
        "I want to run this in federatd mode and use mlp, "
        "CrossEntropy, and FedAvg across the hospital clients.",
        "federated",
    ),
    (
        "Okay so basically we need real FeatureCloud clients in different "
        "organizations, data stays local, only model updates are shared.",
        "federated",
    ),
    (
        "not simulation, not one machine — actual distributed clients "
        "connected through the controller",
        "federated",
    ),
    ("pls start federat learning now", "federated"),
    ("option: real FL with clients", "federated"),
    (
        "My thesis setup needs training on separate nodes where each client "
        "has its own folder and we aggregate after local updates.",
        "federated",
    ),
    ("fed learning w/ 4 clients", "federated"),
    ("run across clients (real)", "federated"),
    (
        "Use the mode where each partner trains locally and we only "
        "exchange weights, not the raw datasets.",
        "federated",
    ),
    ("I want feder... with multiple clients", "federated"),
    # Simulation (20)
    ("Run the app in simulation mode.", "simulation"),
    ("I need simulation for local testing.", "simulation"),
    ("Fake multi-client run on my laptop first.", "simulation"),
    ("I want to dry-run federated locally before real clients.", "simulation"),
    ("sim", "simulation"),
    ("simulation", "simulation"),
    ("simualtion", "simulation"),
    ("simul", "simulation"),
    ("local fake clients please", "simulation"),
    (
        "I want to run this in simualtion and use my sample data folders "
        "to pretend there are several clients.",
        "simulation",
    ),
    (
        "Before contacting real hospitals I just want to emulate multi-client "
        "federated learning on one PC.",
        "simulation",
    ),
    (
        "not real federated yet — just simulate clients locally so I can "
        "debug the config",
        "simulation",
    ),
    ("mock FL run on my machine", "simulation"),
    ("dry run / simulated clients", "simulation"),
    (
        "Can we do a simulated federated experiment where client_dirs are "
        "local folders under data/, no real remote clients?",
        "simulation",
    ),
    ("test multi-client without deploying", "simulation"),
    ("i need sim mode for quick check", "simulation"),
    (
        "Long story short: I want federation behavior but everything should "
        "stay on localhost with fake clients.",
        "simulation",
    ),
    ("simulatn mode pls", "simulation"),
    ("run as if there are clients, but locally", "simulation"),
    # Centralized (20)
    ("Use centralized training on one machine.", "centralized"),
    ("I want centralized mode.", "centralized"),
    ("Train everything on a single server, no federation.", "centralized"),
    ("One dataset on one machine only.", "centralized"),
    ("central", "centralized"),
    ("centralized", "centralized"),
    ("centralised", "centralized"),
    ("centr", "centralized"),
    ("single machine training", "centralized"),
    (
        "I want to run this in centraliyed mode and use one data_dir, "
        "mlp model, and normal training — no clients.",
        "centralized",
    ),
    (
        "Just train on my laptop with one dataset folder. No multi-client, "
        "no simulation of clients, plain centralized DL.",
        "centralized",
    ),
    (
        "not federated and not simulation — classic single-node training "
        "on all data together",
        "centralized",
    ),
    ("one server, one dataset", "centralized"),
    ("no FL, just normal training", "centralized"),
    (
        "Please configure the app for centralized execution because we only "
        "have one machine and one shared dataset path.",
        "centralized",
    ),
    ("central training please", "centralized"),
    ("i dont need clients, just train", "centralized"),
    (
        "We are not doing distributed learning. Put everything in one place "
        "and train end-to-end on that single machine.",
        "centralized",
    ),
    ("centrallized mode", "centralized"),
    ("single-node / non-federated run", "centralized"),
]
