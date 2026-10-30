# Test 2 — GUI workflow, federated

**Goal:** Run the whole GUI path once and see that it reaches a federated config.

## Start

```powershell
& "C:\UH\Thesis\FC\fc-deep-learning-env-zenbook\Scripts\Activate.ps1"
cd "C:\UH\Thesis\FC\fc-deep-learning-master\fc-deep-learning-master"
python -m tools.intake.intake_dspy.cli
```

Do not add `--no-gui`. Leave `FC_DSPY_RECOMPILE` unset. The two intake agents compile. Every agent after `intake_process.json` loads its saved program.

## Window 1 — "Please enter you prompt"

Paste this, then click **Next**:

```text
Run this as federated learning. Two hospitals keep their own data and only share model weights.
Hospital A is sample_data/c1 and hospital B is sample_data/c2.
Inside each client, the training file is train.npz and the test file is test.npz.
The fold directories are in a folder named data.
Use 5 communication rounds, 10 classes, and the built-in FedAvg aggregator.
Train with BasicTrainer, load images with ImageLoader, and use CrossEntropyLoss.
Keep verbose false, train batch 32, test batch 32, 1 local epoch, learning rate 0.1, and batch_count 1.
```

Wait until the terminal prints `intake_briefs.json`. The second window opens after that.

## Window 2 — "Please enter your prompt for model section"

Paste this:

```text
Build a network for 28x28 images.
The constructor takes in_features input channels and n_classes outputs.
number of input is 1 and number of class is 10.

Conv2d: in_features to 10, kernel 5, no padding.
MaxPool2d(2), then ReLU.
Conv2d: 10 to 20, kernel 5, no padding.
Dropout2d.
MaxPool2d(2), then ReLU.
Flatten to 320.
Linear: 320 to 50, then ReLU.
Dropout.
Linear: 50 to n_classes.
Finish the forward pass with log_softmax.
```

The radios start on **No**. Set them before **Start**:

1. `Create a new deep learning architecture model?` — **Yes**
2. `New Architecture Name` — `test2_cnn`
3. `Save New architecture` — **Yes**
4. Click **Start**

## After the windows

No further text is requested from you until the end. The terminal uses `intake_process.json`:

- `execution_mode_dspy` selects the federated workflow.
- `dataset_dspy` receives the `sample_data/c1` and `sample_data/c2` sentence.
- `hyper_params_dspy` receives the rounds, classes, and FedAvg sentence.
- `new_architecture` is `Yes`, so the model agent receives the full layer description, from the first `Conv2d` through `log_softmax`.
- The model is saved as `plugins\models\test2_cnn.py`.
- `model_dspy`, `trainer_dspy`, and `train_config_dspy` receive their own sentences.
- The federated config is written to `tools\config_builder\output\federated\federated_config.yml`.

The last question is in the terminal:

```text
Are you sure you want to run this command? (yes/no)
```

Type `yes` to start FeatureCloud, or `no` to stop after the config file exists.

A line that says `Empty intake key` or `Incompatible intake choices` means `intake_process.json` was not written and the later agents did not start. If the terminal says `cannot save model name 'test2_cnn'`, that name is already under `plugins\models`; run again with a different name.
