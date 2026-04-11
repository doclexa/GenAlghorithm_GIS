#!/usr/bin/env python3
"""
Строит графики по CSV из compare_bf_ga_skv621.py (matplotlib, без pandas).

Запуск из корня GenAlghorithm_GIS:
  python experiments/plot_experiment_results.py --input-dir outputs/experiments/skv621_bf_ga
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
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


def plot_benchmark(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    methods = [r["method"] for r in rows]
    qs = [float(r["Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(range(len(methods)), qs, color=colors[: len(methods)])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Q (calc_quality_score)")
    ax.set_title("Сравнение GA и bruteforce (skv621): итоговое качество")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{qs[i]:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_stability(rows: list[dict[str, str]], out_png: Path) -> None:
    if not rows:
        return
    by_shift: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"bf": [], "ga": []})
    for r in rows:
        sf = r["shift_frac"]
        q = float(r["Q"])
        if r["method"] in ("bf_full", "bf_capped"):
            by_shift[sf]["bf"].append(q)
        elif r["method"] == "ga":
            by_shift[sf]["ga"].append(q)
    shifts = sorted(by_shift.keys(), key=lambda x: float(x))
    bf_means = [statistics.mean(by_shift[s]["bf"]) for s in shifts]
    ga_means = [statistics.mean(by_shift[s]["ga"]) for s in shifts]
    ga_stds = [
        statistics.stdev(by_shift[s]["ga"]) if len(by_shift[s]["ga"]) > 1 else 0.0 for s in shifts
    ]
    xs = [float(s) for s in shifts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, bf_means, "s-", color="#3498db", label="Bruteforce (BF в CSV)", markersize=8)
    ax.errorbar(xs, ga_means, yerr=ga_stds, fmt="o-", color="#2ecc71", capsize=4, label="GA (среднее ± std по сидам)")
    ax.set_xlabel("Доля сдвига границ (translate_bounds)")
    ax.set_ylabel("Q")
    ax.set_title("Устойчивость к сдвигу границ: BF vs GA (skv621)")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    stab = read_csv(d / "stability_skv621.csv")
    if bench:
        plot_benchmark(bench, d / "plot_benchmark_Q.png")
        print(f"Записано: {d / 'plot_benchmark_Q.png'}")
    if stab:
        plot_stability(stab, d / "plot_stability_Q.png")
        print(f"Записано: {d / 'plot_stability_Q.png'}")
    if not bench and not stab:
        print(f"Нет CSV в {d}")


if __name__ == "__main__":
    main()
