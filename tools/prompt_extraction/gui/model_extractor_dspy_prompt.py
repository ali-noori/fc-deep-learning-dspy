"""Prompt window for model_extractor_dspy. It only collects the model message."""

from __future__ import annotations

from dataclasses import dataclass

import tkinter as tk
from tkinter import messagebox


WINDOW_TITLE = "Please enter your prompt for model section"
EMPTY_PROMPT_MESSAGE = "Please enter a prompt before continuing."
PROMPT_HINT = (
    "Give a model name or the layer description, the number of input features, "
    "and the number of classes.\n"
    "Numbers may be digits or words.\n"
    "\n"
    "Example:\n"
    "efficientnet_b0 and number of input is 1 and number of class is ten."
)


def cleaned_prompt(raw: str) -> str:
    return raw.strip()


@dataclass(frozen=True)
class ModelPrompt:
    prompt: str
    new_architecture: str
    new_architecture_name: str
    save_new_architecture: str


class ModelExtractorDspyPromptWindow:
    def __init__(self) -> None:
        self.result: ModelPrompt | None = None
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="white")
        self.root.geometry("920x560")
        self.root.minsize(780, 420)

        blank = tk.PhotoImage(width=1, height=1)
        self.root.iconphoto(False, blank)
        self._blank_icon = blank

        frame = tk.Frame(
            self.root,
            bg="white",
            highlightbackground="#d0d0d0",
            highlightthickness=1,
        )
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        self.hint = tk.Label(
            frame,
            text=PROMPT_HINT,
            bg="white",
            fg="#666666",
            justify="left",
            anchor="w",
            wraplength=700,
            font=("Segoe UI", 9),
        )
        self.hint.pack(fill="x", padx=12, pady=(12, 0))

        self.text = tk.Text(
            frame,
            wrap="word",
            bg="white",
            fg="#222222",
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground="#d0d0d0",
            highlightcolor="#b0b0b0",
            font=("Segoe UI", 11),
            padx=8,
            pady=8,
        )
        self.text.pack(fill="both", expand=True, padx=12, pady=(8, 8))

        bar = tk.Frame(frame, bg="white")
        bar.pack(fill="x", padx=12, pady=(0, 12))
        bar.columnconfigure(1, weight=1)

        create_label = tk.Label(
            bar,
            text="Create a new deep learning architecture model?",
            bg="white",
            fg="#222222",
            justify="left",
            anchor="w",
            wraplength=250,
            font=("Segoe UI", 9),
        )
        create_label.grid(row=0, column=0, sticky="w", padx=(0, 16))

        self.name_label = tk.Label(
            bar,
            text="New Architecture Name",
            bg="white",
            fg="#222222",
            justify="left",
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.name_label.grid(row=0, column=1, sticky="w", padx=(0, 16))

        radio_row = tk.Frame(bar, bg="white")
        radio_row.grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(6, 0))
        self.create_new = tk.StringVar(value="No")
        for label in ("Yes", "No"):
            tk.Radiobutton(
                radio_row,
                text=label,
                variable=self.create_new,
                value=label,
                command=self._sync_name_field,
                bg="white",
                activebackground="white",
                fg="#222222",
                selectcolor="white",
                font=("Segoe UI", 10),
                highlightthickness=0,
            ).pack(side="left", padx=(0, 8))

        self.name_entry = tk.Entry(
            bar,
            relief="solid",
            bd=1,
            bg="white",
            fg="#222222",
            disabledbackground="#f2f2f2",
            disabledforeground="#888888",
            highlightthickness=1,
            highlightbackground="#d0d0d0",
            highlightcolor="#b0b0b0",
            font=("Segoe UI", 11),
        )
        self.name_entry.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=(6, 0))

        self.save_label = tk.Label(
            bar,
            text="Save New architecture",
            bg="white",
            fg="#222222",
            justify="left",
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.save_label.grid(row=0, column=2, sticky="w", padx=(0, 16))

        save_row = tk.Frame(bar, bg="white")
        save_row.grid(row=1, column=2, sticky="w", padx=(0, 16), pady=(6, 0))
        self.save_new = tk.StringVar(value="No")
        self._save_radios: list[tk.Radiobutton] = []
        for label in ("Yes", "No"):
            button = tk.Radiobutton(
                save_row,
                text=label,
                variable=self.save_new,
                value=label,
                bg="white",
                activebackground="white",
                fg="#222222",
                selectcolor="white",
                font=("Segoe UI", 10),
                highlightthickness=0,
            )
            button.pack(side="left", padx=(0, 8))
            self._save_radios.append(button)

        start_button = tk.Button(
            bar,
            text="Start",
            command=self.start,
            bg="#2e9b4f",
            fg="white",
            activebackground="#278845",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        start_button.grid(row=1, column=3, sticky="e", pady=(6, 0))

        self._sync_name_field()
        self.root.bind("<Configure>", self._fit_hint)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.text.focus_set()

    def _fit_hint(self, _event: tk.Event | None = None) -> None:
        if not self.root.winfo_exists() or not self.hint.winfo_exists():
            return
        width = max(self.root.winfo_width() - 80, 280)
        self.hint.configure(wraplength=width)

    def _sync_name_field(self) -> None:
        enabled = self.create_new.get() == "Yes"
        self.name_entry.configure(state="normal" if enabled else "disabled")
        self.name_label.configure(fg="#222222" if enabled else "#888888")
        radio_state = "normal" if enabled else "disabled"
        radio_color = "#222222" if enabled else "#888888"
        self.save_label.configure(fg=radio_color)
        for button in self._save_radios:
            button.configure(state=radio_state, fg=radio_color)

    def start(self) -> None:
        value = cleaned_prompt(self.text.get("1.0", "end"))
        if not value:
            messagebox.showwarning(WINDOW_TITLE, EMPTY_PROMPT_MESSAGE, parent=self.root)
            return
        self.result = ModelPrompt(
            prompt=value,
            new_architecture=self.create_new.get(),
            new_architecture_name=self.name_entry.get().strip(),
            save_new_architecture=self.save_new.get(),
        )
        self.root.destroy()

    def show(self) -> ModelPrompt | None:
        self.root.mainloop()
        return self.result


def ask_model_extractor_dspy_prompt() -> ModelPrompt | None:
    """Show the model_extractor_dspy window.

    Returns the prompt and the three architecture choices when the user clicks Start.
    Returns None when the user closes the window.
    """
    return ModelExtractorDspyPromptWindow().show()
