#!/usr/bin/env python3
"""
Compute per-client/per-split accuracy from FeatureCloud output CSV files.

Default behavior scans ./data/tests for:
- extracted outputs containing y_pred.csv / y_test.csv
- zip bundles like results_test_*_client_*.zip that contain those CSV files

By default only the **last 3 test runs** are included (grouped from FeatureCloud zip names/mtimes).
Use `--all` or `--last-n 0` to report every run.

Usage examples:
  python ali_test.py
  python ali_test.py --last-n 5
  python ali_test.py --all
  python ali_test.py --base-dir "C:/FC/fc-deep-learning-master/fc-deep-learning-master/data/tests"
  python ali_test.py --pred-name y_pred.csv --target-name y_test.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Set


@dataclass
class AccuracyResult:
    client: str
    split: str
    n: int
    correct: int
    accuracy: float
    source: str


def read_single_column_csv_bytes(raw: bytes) -> List[str]:
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []
    # OLD: assume fixed header names.
    # NEW: generic parser; drop first row only if it looks like a header.
    first = rows[0]
    data_rows = rows[1:] if first and first[0].lower() in {"y_pred", "y_true", "target", "label"} else rows
    out: List[str] = []
    for r in data_rows:
        if not r:
            continue
        out.append(r[0].strip())
    return out


def infer_client_and_split_from_path(path_text: str) -> Tuple[str, str]:
    p = path_text.replace("\\", "/")
    client = "unknown-client"
    split = "unknown-split"

    m_client = re.search(r"/client[_-]?(\d+)\b", p, flags=re.IGNORECASE)
    if m_client:
        client = f"client_{m_client.group(1)}"

    # OLD: only inspect exact 'output/<split>/' layout.
    # NEW: detect generic '/<digit>/y_pred.csv' style as used by split folders.
    m_split = re.search(r"/(\d+)/[^/]*y_(?:pred|test)\.csv$", p, flags=re.IGNORECASE)
    if m_split:
        split = m_split.group(1)
    return client, split


def load_pairs_from_zip(zip_path: Path, pred_name: str, target_name: str) -> List[AccuracyResult]:
    results: List[AccuracyResult] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        preds = [n for n in names if n.lower().endswith("/" + pred_name.lower()) or Path(n).name.lower() == pred_name.lower()]
        targets = [n for n in names if n.lower().endswith("/" + target_name.lower()) or Path(n).name.lower() == target_name.lower()]

        # Build map by folder key
        target_by_folder: Dict[str, str] = {str(Path(t).parent).replace("\\", "/"): t for t in targets}
        for pred in preds:
            folder = str(Path(pred).parent).replace("\\", "/")
            target = target_by_folder.get(folder)
            if not target:
                continue

            pred_vals = read_single_column_csv_bytes(zf.read(pred))
            true_vals = read_single_column_csv_bytes(zf.read(target))
            n = min(len(pred_vals), len(true_vals))
            if n == 0:
                continue
            correct = sum(1 for i in range(n) if pred_vals[i] == true_vals[i])
            acc = correct / n

            client, split = infer_client_and_split_from_path(folder + "/" + pred_name)
            # fallback from zip filename
            if client == "unknown-client":
                m = re.search(r"client[_-]?(\d+)", zip_path.name, flags=re.IGNORECASE)
                if m:
                    client = f"client_{m.group(1)}"

            results.append(
                AccuracyResult(
                    client=client,
                    split=split,
                    n=n,
                    correct=correct,
                    accuracy=acc,
                    source=str(zip_path),
                )
            )
    return results


def read_single_column_csv_file(path: Path) -> List[str]:
    return read_single_column_csv_bytes(path.read_bytes())


def load_pairs_from_extracted(base_dir: Path, pred_name: str, target_name: str) -> List[AccuracyResult]:
    results: List[AccuracyResult] = []
    pred_files = list(base_dir.rglob(pred_name))
    for pred in pred_files:
        target = pred.with_name(target_name)
        if not target.exists():
            continue
        pred_vals = read_single_column_csv_file(pred)
        true_vals = read_single_column_csv_file(target)
        n = min(len(pred_vals), len(true_vals))
        if n == 0:
            continue
        correct = sum(1 for i in range(n) if pred_vals[i] == true_vals[i])
        acc = correct / n
        client, split = infer_client_and_split_from_path(str(pred))
        results.append(
            AccuracyResult(
                client=client,
                split=split,
                n=n,
                correct=correct,
                accuracy=acc,
                source=str(pred),
            )
        )
    return results


def parse_results_test_id(zip_name: str) -> Optional[int]:
    """FeatureCloud saves zips like results_test_<id>_client_<n>_....zip"""
    m = re.search(r"results_test_(\d+)_", zip_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def cluster_zip_files_into_runs(zips: List[Path], gap_seconds: float) -> List[List[Path]]:
    """
    Group zip outputs into "runs": same FeatureCloud results_test_<id>, files whose mtimes are
    within gap_seconds of the previous file in time order (per id). Then sort runs by newest file first.
    """
    by_tid: Dict[Optional[int], List[Path]] = defaultdict(list)
    for z in zips:
        by_tid[parse_results_test_id(z.name)].append(z)

    all_clusters: List[List[Path]] = []
    for tid, paths in by_tid.items():
        if tid is None:
            for z in paths:
                all_clusters.append([z])
            continue
        paths_sorted = sorted(paths, key=lambda p: p.stat().st_mtime)
        cluster: List[Path] = []
        for z in paths_sorted:
            if not cluster:
                cluster = [z]
            elif z.stat().st_mtime - cluster[-1].stat().st_mtime <= gap_seconds:
                cluster.append(z)
            else:
                all_clusters.append(cluster)
                cluster = [z]
        if cluster:
            all_clusters.append(cluster)

    all_clusters.sort(key=lambda c: max(p.stat().st_mtime for p in c), reverse=True)
    return all_clusters


def max_mtime_in_tree(path: Path) -> float:
    newest = path.stat().st_mtime
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass
    return newest


def deduplicate(results: Iterable[AccuracyResult]) -> List[AccuracyResult]:
    # Keep one result per (client, split, n, correct) to avoid zip+extracted duplicates.
    seen = set()
    out: List[AccuracyResult] = []
    for r in results:
        key = (r.client, r.split, r.n, r.correct)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def print_report(results: List[AccuracyResult]) -> None:
    if not results:
        print("No matching prediction/target CSV pairs were found.")
        return

    results_sorted = sorted(results, key=lambda r: (r.client, r.split, r.source))
    print("Per-client / per-split accuracy")
    print("-" * 80)
    print(f"{'Client':<14} {'Split':<8} {'N':>8} {'Correct':>8} {'Accuracy':>10}  Source")
    print("-" * 80)
    for r in results_sorted:
        print(f"{r.client:<14} {r.split:<8} {r.n:>8} {r.correct:>8} {r.accuracy:>9.4f}  {r.source}")
    print("-" * 80)

    total_n = sum(r.n for r in results_sorted)
    total_correct = sum(r.correct for r in results_sorted)
    weighted_acc = total_correct / total_n if total_n else 0.0
    mean_acc = statistics.mean(r.accuracy for r in results_sorted)
    print(f"Overall weighted accuracy: {weighted_acc:.4f} ({total_correct}/{total_n})")
    print(f"Mean of row accuracies : {mean_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy from y_pred.csv / y_test.csv results.")
    parser.add_argument(
        "--base-dir",
        default="data/tests",
        help="Base directory containing test output zips or extracted result folders (default: data/tests).",
    )
    parser.add_argument("--pred-name", default="y_pred.csv", help="Prediction CSV filename (default: y_pred.csv).")
    parser.add_argument("--target-name", default="y_test.csv", help="Target CSV filename (default: y_test.csv).")
    parser.add_argument(
        "--last-n",
        type=int,
        default=3,
        metavar="N",
        help="Only include the N most recent test runs (default: 3). Use 0 for all runs (same as --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include every zip and full recursive scan of base-dir (ignores --last-n).",
    )
    parser.add_argument(
        "--run-gap-seconds",
        type=float,
        default=180.0,
        help=(
            "When grouping FeatureCloud zips into one run, max seconds between newest file in run "
            "and another zip with the same results_test_<id> (default: 180)."
        ),
    )
    parser.add_argument(
        "--include-all-extracted",
        action="store_true",
        help=(
            "When using --last-n, still scan all extracted CSV pairs under base-dir (may include old runs). "
            "Default: only scan top-level subfolders picked by recency (see below)."
        ),
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return

    show_all = args.all or args.last_n <= 0
    all_results: List[AccuracyResult] = []

    all_zips = [z for z in base_dir.rglob("*.zip") if z.is_file()]
    zips_to_process: Optional[Set[Path]] = None

    if not show_all and all_zips:
        runs = cluster_zip_files_into_runs(all_zips, args.run_gap_seconds)
        keep = runs[: max(0, args.last_n)]
        zips_to_process = {z for run in keep for z in run}
        print(
            f"Restricting to the last {len(keep)} test run(s) by zip clustering "
            f"(gap={args.run_gap_seconds}s, {len(zips_to_process)} zip file(s))."
        )

    if show_all:
        all_results.extend(load_pairs_from_extracted(base_dir, args.pred_name, args.target_name))
        zip_iter = all_zips
    elif args.include_all_extracted:
        all_results.extend(load_pairs_from_extracted(base_dir, args.pred_name, args.target_name))
        zip_iter = sorted(zips_to_process) if zips_to_process else []
    elif zips_to_process is not None:
        # Extracted layout: prefer one folder per run under base_dir (by newest file mtime).
        subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
        subdirs.sort(key=max_mtime_in_tree, reverse=True)
        picked = subdirs[: max(0, args.last_n)]
        if picked:
            print(f"Also scanning {len(picked)} most recently modified top-level subfolder(s) for extracted CSVs.")
        for sub in picked:
            all_results.extend(load_pairs_from_extracted(sub, args.pred_name, args.target_name))
        zip_iter = sorted(zips_to_process)
    else:
        # No zips: fall back to full extracted scan (same as legacy behavior).
        all_results.extend(load_pairs_from_extracted(base_dir, args.pred_name, args.target_name))
        zip_iter = []

    for z in zip_iter:
        try:
            all_results.extend(load_pairs_from_zip(z, args.pred_name, args.target_name))
        except zipfile.BadZipFile:
            # Skip non-zip files with .zip extension.
            continue

    final_results = deduplicate(all_results)
    print_report(final_results)


if __name__ == "__main__":
    main()
