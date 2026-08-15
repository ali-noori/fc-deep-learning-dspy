"""
Where the "smart writing" happens.

Big picture (like explaining to a friend):
  • You give plain-English directions ("make a tiny CNN for 28×28 pictures…").
  • DSPy is a helper library that asks a big language model (on the internet or your
    own computer) to write Python code for a `torch.nn.Module`.
  • We show the model a few finished examples—like showing someone three solved
    homework sheets before they try a new problem. That is called "few-shot."
  • Chain-of-Thought means the model is nudged to think step-by-step before it writes
    the final code, which usually comes out a bit neater.

If the cloud helpers are sleepy or busy, we fall back to Ollama on your machine.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# DSPy is an optional dependency: the rest of the project can still import paths
# even when DSPy is missing, until someone actually runs this tool.
try:
    import dspy

    HAS_DSPY = True
except Exception:
    dspy = None
    HAS_DSPY = False


# Ollama normally listens here on your PC unless you change the environment variable.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1"

# Provider API keys: read from tools/keys_secrets.py (gitignored)
try:
    from tools.keys_secrets import (
        GROQ_API_KEY,
        GEMINI_API_KEY,
        COHERE_API_KEY,
        MISTRAL_API_KEY,
    )
except ImportError:
    GROQ_API_KEY = ""
    GEMINI_API_KEY = ""
    COHERE_API_KEY = ""
    MISTRAL_API_KEY = ""


@dataclass(frozen=True)
class ArchitectureDescription:
    """Everything we need after one successful generation."""

    # The actual Python source the model returned (cleaned a little).
    code: str
    # Which family answered: "groq", "mistral", "ollama", …
    backend: str
    # The pretty model name for humans (for example `codestral-latest`).
    model: str
    # Only filled for Ollama; cloud routes do not need a local URL.
    base_url: str | None


def get_ollama_base_url() -> str:
    """Read `FC_DSPY_OLLAMA_BASE_URL` if set, else use the normal home-computer default."""
    return os.environ.get("FC_DSPY_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def get_ollama_model() -> str:
    """Read `FC_DSPY_OLLAMA_MODEL` if set, else use our usual Ollama model nickname."""
    return os.environ.get("FC_DSPY_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


# "Teacher examples": each item is (student question text, perfect answer code).
# DSPy shows these pairs to the language model so it learns your style—like a spelling
# test practice sheet before the real test. The list is long because CNN and MLP ideas
# need several patterns (pooling, softmax, batch norm, etc.).
FEWSHOT_PAIRS: list[tuple[str, str]] = [
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->10, kernel=5, no padding
MaxPool2d(2) -> ReLU
Conv2d: 10->20, kernel=5, no padding
Dropout2d
MaxPool2d(2) -> ReLU
Flatten to 320
Linear: 320->50 -> ReLU
Dropout
Linear: 50->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->20, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 20->50, kernel=3, padding=1
MaxPool2d(2) -> ReLU
AdaptiveAvgPool2d(4x4)
Flatten to 800
Linear: 800->128 -> ReLU
Dropout1d(0.2)
Linear: 128->n_classes
no softmax output""",
        """import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(50 * 4 * 4, 128)
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return x""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->20, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 20->50, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 50->100, kernel=5, padding=1
MaxPool2d(2) -> ReLU
AdaptiveAvgPool2d(4x4)
Flatten to 1600
Linear: 1600->256 -> ReLU
Dropout1d(0.2)
Linear: 256->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(100 * 4 * 4, 256)
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2(x)))
        x = self.act(self.pool(self.conv3(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->32, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 32->64, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 64->128, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 128->256, kernel=3, padding=1
BatchNorm2d
ReLU
AdaptiveAvgPool2d(2x2)
Flatten to 1024
Linear: 1024->512 -> ReLU
Dropout(0.4)
Linear: 512->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.drop = nn.Dropout(0.4)
        self.fc_out = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.act(self.bn4(self.conv4(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an MLP architecture for multiclass classification with n_classes classes and in_features input features.

Linear: in_features->64 -> SiLU
Dropout(0.2)
Linear: 64->32 -> SiLU
Dropout(0.2)
Linear: 32->n_classes
no softmax output""",
        """import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, n_classes)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.act(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        return x""",
    ),
    (
        """Create an MLP architecture for multiclass classification with n_classes classes and in_features input features.

Linear: in_features->128 -> BatchNorm1d -> ReLU
Dropout(0.25)
Linear: 128->n_classes
no softmax""",
        """import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, n_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.fc_out(x)
        return x""",
    ),
]


def _strip_code_fence(text: str) -> str:
    """If the model wrapped code in ``` fences, peel them off so we keep plain Python."""
    cleaned = text.strip()
    # Many chat-style models return ```python ... ``` even when we asked for bare code.
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # If the fences are only half there, strip the leftover backticks by hand.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _extract_python_code(text: str) -> str:
    """Keep only the part that looks like real Python, starting at the first import."""
    cleaned = _strip_code_fence(text)
    # Sometimes the model adds chit-chat before `import torch`; we skip that noise.
    import_match = re.search(r"(?m)^(?:import|from)\s+", cleaned)
    if import_match:
        cleaned = cleaned[import_match.start() :]
    return cleaned.strip()


if HAS_DSPY:

    class ArchitectureSignature(dspy.Signature):
        """
        The "worksheet blanks" DSPy fills in.

        `prompt` is what you typed. `architecture_code` must become importable Python
        with a class `Model` that PyTorch can load.
        """

        prompt = dspy.InputField(
            desc=(
                "Natural-language model architecture description. May include image size, "
                "in_features, n_classes, ordered layer list, activations, flatten size, and "
                "softmax / no-softmax output rule."
            )
        )
        architecture_code = dspy.OutputField(
            desc=(
                "Python source code only. Define exactly one class Model(nn.Module) with "
                "__init__(self, n_classes, in_features) and forward(self, x). Follow the "
                "demo Prompt/Architecture pairs for naming, layer order, flatten style, "
                "dropout handling, and softmax behavior."
            )
        )

    def _build_demos() -> list:
        """Turn every FEWSHOT pair into a DSPy example the model can study."""
        return [
            dspy.Example(prompt=p, architecture_code=c).with_inputs("prompt")
            for p, c in FEWSHOT_PAIRS
        ]

    class ArchitectureGenerator(dspy.Module):
        """
        The worker that actually talks to the language models.

        It keeps one Chain-of-Thought "student" and a backpack of solved examples.
        Then it tries each backend in order—if one throws an error, it simply moves
        to the next helper instead of giving up.
        """

        def __init__(
            self,
            ollama_model: str | None = None,
            ollama_base_url: str | None = None,
        ) -> None:
            super().__init__()
            self.ollama_model = ollama_model or get_ollama_model()
            self.ollama_base_url = (ollama_base_url or get_ollama_base_url()).rstrip("/")

            # `dspy.LM` is a small helper that knows how to call each company's API.
            ollama_lm = dspy.LM(
                f"ollama/{self.ollama_model}", api_base=self.ollama_base_url
            )

            # Try the fast cloud brains first; save Ollama for last (your own computer).
            # Each triple is: short name, model nickname people recognize, LM object.
            self._backend_chain: list[tuple[str, str, dspy.LM]] = [
                (
                    "groq",
                    "mixtral-8x7b-32768",
                    dspy.LM("groq/mixtral-8x7b-32768", api_key=GROQ_API_KEY),
                ),
                (
                    "gemini",
                    "gemini-1.5-pro",
                    dspy.LM("gemini/gemini-1.5-pro", api_key=GEMINI_API_KEY),
                ),
                (
                    "cohere",
                    "command-r-plus",
                    dspy.LM("cohere/command-r-plus", api_key=COHERE_API_KEY),
                ),
                (
                    "mistral",
                    "codestral-latest",
                    dspy.LM("mistral/codestral-latest", api_key=MISTRAL_API_KEY),
                ),
                ("ollama", self.ollama_model, ollama_lm),
            ]

            # Default global setting (DSPy likes having *some* LM configured up front).
            dspy.configure(lm=self._backend_chain[0][2])
            # Chain-ofThought = think a little, then answer. Demos = few-shot examples.
            self.generate = dspy.ChainOfThought(ArchitectureSignature)
            self.generate.demos = _build_demos()

        def forward(self, prompt: str) -> ArchitectureDescription:
            """Return cleaned code plus which backend won the race."""
            request = prompt.strip()
            last_err: BaseException | None = None
            for backend, model_label, lm in self._backend_chain:
                try:
                    # Passing `lm=` makes sure this call really uses that provider,
                    # instead of accidentally reusing the last one in memory.
                    result = self.generate(prompt=request, lm=lm)
                    code = _extract_python_code(result.architecture_code)
                    base_url = self.ollama_base_url if backend == "ollama" else None
                    return ArchitectureDescription(
                        code=code,
                        backend=backend,
                        model=model_label,
                        base_url=base_url,
                    )
                except BaseException as exc:
                    # Wrong key? Model asleep? Network hiccup? Try the next friend.
                    last_err = exc
                    continue

            raise RuntimeError(
                "All language-model backends failed (Groq, Gemini, Cohere, Mistral, Ollama). "
                f"Last error: {last_err!r}"
            ) from last_err

else:

    class ArchitectureGenerator:
        """Tiny stub so imports never explode when DSPy is missing from the venv."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("DSPy is not installed. Install dspy-ai to use model_plugin_dspy_ali.")
