"""Prompt window for intake_dspy. It only collects the intake message."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


WINDOW_TITLE = "Please enter you prompt"
EMPTY_PROMPT_MESSAGE = "Please enter a prompt before continuing."
PROMPT_HINT = (
    "Example:\n"
    "Federated.\n"
    "Clients sample_data/c1 and sample_data/c2. train.npz, test.npz, split folder data.\n"
    "5 rounds, 10 classes, FedAvg.\n"
    "BasicTrainer, ImageLoader, CrossEntropyLoss.\n"
    "verbose false, batch 32, test batch 32, 1 epoch, lr 0.1, batch_count 1."
)


def cleaned_prompt(raw: str) -> str:
    return raw.strip()


class IntakeDspyPromptWindow:
    def __init__(self) -> None:
        self.result: str | None = None
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="white")
        self.root.geometry("760x560")
        self.root.minsize(520, 320)

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

        hint = tk.Label(
            frame,
            text=PROMPT_HINT,
            bg="white",
            fg="#666666",
            justify="left",
            anchor="w",
            wraplength=700,
            font=("Segoe UI", 9),
        )
        hint.pack(fill="x", padx=12, pady=(12, 0))

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

        button_row = tk.Frame(frame, bg="white")
        button_row.pack(fill="x", padx=12, pady=(0, 12))
        next_button = tk.Button(
            button_row,
            text="Next",
            command=self.next,
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
        next_button.pack(side="right")

        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.text.focus_set()

    def next(self) -> None:
        value = cleaned_prompt(self.text.get("1.0", "end"))
        if not value:
            messagebox.showwarning(WINDOW_TITLE, EMPTY_PROMPT_MESSAGE, parent=self.root)
            return
        self.result = value
        self.root.destroy()

    def show(self) -> str | None:
        self.root.mainloop()
        return self.result


def ask_intake_dspy_prompt() -> str | None:
    """Show the intake_dspy window.

    Returns the text when the user clicks Next.
    Returns None when the user closes the window.
    """
    return IntakeDspyPromptWindow().show()
