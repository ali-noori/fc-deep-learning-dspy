#!/usr/bin/env python3
"""
Accuracy from y_pred.csv / y_test.csv pairs inside FeatureCloud result zips.

  python tools/report/report_accuracy.py
      Use the newest test under <project>/data/tests (all client zips
      that share that test id).

  python tools/report/report_accuracy.py --path path/to/results.zip
      Use only that zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

PRED_NAME = "y_pred.csv"
TARGET_NAME = "y_test.csv"
# tools/report/this file → project root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "data" / "tests"


def first_column(raw: bytes) -> list[str]:
    """Turn CSV bytes into a list of labels (skip a header row if present)."""
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows:
        return []
    header = {"y_pred", "y_true", "target", "label"}
    start = 1 if rows[0] and rows[0][0].strip().lower() in header else 0
    return [row[0].strip() for row in rows[start:] if row]


def pair_accuracy(pred_vals: list[str], true_vals: list[str]) -> tuple[int, int, float]:
    """Count matches on the shared length of the two columns."""
    n = min(len(pred_vals), len(true_vals))
    if n == 0:
        return 0, 0, 0.0
    correct = sum(1 for i in range(n) if pred_vals[i] == true_vals[i])
    return n, correct, correct / n


def folder_key(zip_name: str) -> str:
    """Folder inside the zip that contains a CSV ('.' if the file is at the zip root)."""
    parent = Path(zip_name.replace("\\", "/")).parent
    text = str(parent).replace("\\", "/")
    return "." if text in {"", "."} else text


def pairs_in_zip(zip_path: Path) -> list[tuple[str, int, int, float]]:
    """Find every y_pred.csv that has a y_test.csv next to it and score that pair."""
    rows: list[tuple[str, int, int, float]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        targets = {
            folder_key(n): n
            for n in names
            if Path(n).name.lower() == TARGET_NAME
        }
        for name in names:
            if Path(name).name.lower() != PRED_NAME:
                continue
            folder = folder_key(name)
            target_name = targets.get(folder)
            if not target_name:
                continue
            n, correct, acc = pair_accuracy(
                first_column(zf.read(name)),
                first_column(zf.read(target_name)),
            )
            if n == 0:
                continue
            rows.append((folder, n, correct, acc))
    return rows


def test_id_from_name(filename: str) -> str | None:
    """FeatureCloud names zips results_test_<id>_client_....zip — used to group one run."""
    match = re.search(r"results_test_(\d+)_", filename, flags=re.IGNORECASE)
    return match.group(1) if match else None


def latest_run_zips(tests_dir: Path) -> list[Path]:
    """Newest zip in tests_dir, plus siblings that share the same results_test_<id>."""
    zips = sorted(tests_dir.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    zips = [p for p in zips if zipfile.is_zipfile(p)]
    if not zips:
        return []
    newest = zips[0]
    run_id = test_id_from_name(newest.name)
    if run_id is None:
        return [newest]
    return sorted(p for p in zips if test_id_from_name(p.name) == run_id)


def print_zip_report(zip_path: Path) -> tuple[int, int]:
    """Print one zip’s pairs; return totals so the caller can sum several zips."""
    rows = pairs_in_zip(zip_path)
    print(f"\n{zip_path}")
    if not rows:
        print("  No y_pred.csv / y_test.csv pairs found.")
        return 0, 0
    total_n = total_correct = 0
    for folder, n, correct, acc in rows:
        print(f"  {folder:<40}  n={n:<6} correct={correct:<6} accuracy={acc:.4f}")
        total_n += n
        total_correct += correct
    return total_n, total_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy from y_pred.csv / y_test.csv in result zips.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="One result .zip. If omitted, use the last test under data/tests.",
    )
    args = parser.parse_args()

    if args.path is not None:
        zip_path = args.path.expanduser().resolve()
        if not zip_path.is_file() or not zipfile.is_zipfile(zip_path):
            print(f"Not a zip file: {zip_path}")
            return
        zips = [zip_path]
    else:
        tests_dir = DEFAULT_TESTS_DIR
        if not tests_dir.is_dir():
            print(f"Tests folder does not exist: {tests_dir}")
            return
        zips = latest_run_zips(tests_dir)
        if not zips:
            print(f"No zip files found under {tests_dir}")
            return
        print(f"Last test: {len(zips)} zip file(s) under {tests_dir}")

    total_n = total_correct = 0
    for z in zips:
        n, correct = print_zip_report(z)
        total_n += n
        total_correct += correct

    if len(zips) > 1 and total_n:
        print("-" * 40)
        print(f"Overall  n={total_n}  correct={total_correct}  accuracy={total_correct / total_n:.4f}")


if __name__ == "__main__":
    main()
