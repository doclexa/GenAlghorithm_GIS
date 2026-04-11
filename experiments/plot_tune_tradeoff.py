#!/usr/bin/env python3
"""
Scatter: время поиска vs Q по CSV из tune_mkm_gen_hyperparams.py.

  python experiments/plot_tune_tradeoff.py --csv outputs/mkm_gen_tuning_report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from mkm_core import resolve_path  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/mkm_gen_tuning_report.csv")
    p.add_argument("--out", default="", help="PNG (по умолчанию рядом с CSV: *_time_vs_Q.png)")
    args = p.parse_args()
    path = resolve_path(args.csv, PROJECT_ROOT)
    if not path.is_file():
        print(f"Файл не найден: {path}")
        sys.exit(1)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV пуст")
        sys.exit(1)
    t = [float(r["search_time_sec"]) for r in rows]
    q = [float(r["quality_score"]) for r in rows]
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = resolve_path(str(out), PROJECT_ROOT)
    else:
        out = path.with_name(path.stem + "_time_vs_Q.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(t, q, alpha=0.65, c="#2980b9", edgecolors="k", linewidths=0.3, s=40)
    ax.set_xlabel("Время поиска GA, с")
    ax.set_ylabel("Q (качество МКМ)")
    ax.set_title("Случайный поиск гиперпараметров: время vs качество")
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
