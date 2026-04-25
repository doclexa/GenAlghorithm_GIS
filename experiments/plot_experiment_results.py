#!/usr/bin/env python3
"""Строит сравнительные графики по CSV из compare_bf_ga_skv621.py."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _method_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row["method"] for row in rows]


def plot_benchmark_quality(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    methods = _method_labels(rows)
    qs = [float(r["Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if "ga" in method else "#3498db" for method in methods]
    bars = ax.bar(range(len(methods)), qs, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Q (меньше лучше)")
    ax.set_title("Итоговое качество интервальной МКМ")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{qs[i]:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_time_vs_quality(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        x = float(row["time_sec"])
        y = float(row["Q"])
        label = row["method"]
        color = "#2ecc71" if "ga" in label else "#3498db"
        marker = "o" if "ga" in label else "s"
        ax.scatter([x], [y], color=color, marker=marker, s=70)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Q (меньше лучше)")
    ax.set_title("Время vs качество")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_evals_vs_quality(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        x = float(row["evals"])
        y = float(row["Q"])
        label = row["method"]
        color = "#2ecc71" if "ga" in label else "#3498db"
        marker = "o" if "ga" in label else "s"
        ax.scatter([x], [y], color=color, marker=marker, s=70)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Число оценок / операций")
    ax.set_ylabel("Q (меньше лучше)")
    ax.set_title("Операции vs качество")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_interval_delta(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    xs = [0.5 * (float(row["depth_start"]) + float(row["depth_end"])) for row in rows]
    ys = [float(row["delta_local_score"]) for row in rows]
    colors = ["#3498db" if float(row["delta_local_score"]) > 0 else "#2ecc71" for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs, ys, color=colors, width=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Средняя глубина интервала")
    ax.set_ylabel("BF local_score - GA local_score")
    ax.set_title("Где brute force теряет качество относительно GA")
    ax.grid(True, axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="outputs/experiments/skv621_bf_ga")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = resolve_path(args.input_dir, PROJECT_ROOT)
    bench = read_csv(d / "benchmark_skv621.csv")
    interval_delta = read_csv(d / "interval_comparison_bf_matched_vs_ga.csv")
    if bench:
        plot_benchmark_quality(bench, d / "plot_benchmark_Q.png")
        print(f"Записано: {d / 'plot_benchmark_Q.png'}")
        plot_time_vs_quality(bench, d / "plot_time_vs_quality.png")
        print(f"Записано: {d / 'plot_time_vs_quality.png'}")
        plot_evals_vs_quality(bench, d / "plot_evals_vs_quality.png")
        print(f"Записано: {d / 'plot_evals_vs_quality.png'}")
    if interval_delta:
        plot_interval_delta(interval_delta, d / "plot_interval_delta_local_score.png")
        print(f"Записано: {d / 'plot_interval_delta_local_score.png'}")
    if not bench and not interval_delta:
        print(f"Нет CSV в {d}")


if __name__ == "__main__":
    main()
