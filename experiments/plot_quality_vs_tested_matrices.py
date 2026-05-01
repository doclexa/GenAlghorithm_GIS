#!/usr/bin/env python3
"""Отдельный эксперимент: качество vs количество опробованных матриц для bf_N и ga."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    load_mkm_from_las,
    resolve_path,
    split_lithotype_intervals,
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_ga_engine import (  # noqa: E402
    DEFAULT_GA_CXPB,
    DEFAULT_GA_INDPB,
    DEFAULT_GA_MUTPB,
    DEFAULT_GA_PATIENCE,
    DEFAULT_GA_POPULATION_SIZE,
    DEFAULT_GA_TOURNSIZE,
    GAParams,
)
from mkm_interval_optimizer import run_interval_bruteforce, run_interval_ga  # noqa: E402


def build_subdivision_k_matrix(base_k: np.ndarray, subdivisions: int) -> np.ndarray:
    if subdivisions < 1:
        raise ValueError("Количество разбиений должно быть >= 1.")
    base = np.asarray(base_k, dtype=int)
    active_mask = base > 1
    result = np.ones_like(base, dtype=int)
    result[active_mask] = int(subdivisions)
    return result


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "method_family",
        "subdivisions",
        "tested_matrices",
        "quality_score",
        "negative_share",
        "glin_bad_share",
        "coll_bad_share",
        "time_sec",
        "intervals",
        "invalid_count",
        "generations",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_from_rows(rows: list[dict[str, str]] | list[dict[str, object]], out_png: Path) -> None:
    bf_rows = sorted(
        [row for row in rows if str(row["method_family"]) == "bf"],
        key=lambda row: int(row["tested_matrices"]),
    )
    ga_rows = [row for row in rows if str(row["method_family"]) == "ga"]

    fig, ax = plt.subplots(figsize=(9, 5))

    if bf_rows:
        xs = [int(row["tested_matrices"]) for row in bf_rows]
        ys = [float(row["quality_score"]) for row in bf_rows]
        ax.plot(xs, ys, "o-", color="#3498db", label="BF (bf_N)")
        for row in bf_rows:
            ax.annotate(
                str(row["label"]),
                (int(row["tested_matrices"]), float(row["quality_score"])),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=9,
            )

    if ga_rows:
        for row in ga_rows:
            x = int(row["tested_matrices"])
            y = float(row["quality_score"])
            ax.scatter([x], [y], marker="*", s=180, color="#2ecc71", label="GA")
            ax.annotate(
                str(row["label"]),
                (x, y),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=10,
            )

    ax.set_xlabel("Количество опробованных матриц")
    ax.set_ylabel("Q (меньше лучше)")
    ax.set_title("Качество vs количество опробованных матриц")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value)):,}".replace(",", " ")))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    uniq: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in uniq:
            uniq[label] = handle
    ax.legend(uniq.values(), uniq.keys())

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--las", default=DEFAULT_LAS_RELPATH)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument(
        "--subdivisions",
        default="2,3,4,5",
        help="Список N для серий bf_N, где N = число разбиений активных узлов матрицы.",
    )
    parser.add_argument("--output-dir", default="outputs/experiments/skv621_bf_vs_ga_matrices")
    parser.add_argument("--csv", default="", help="Пусто → <output-dir>/quality_vs_tested_matrices.csv")
    parser.add_argument("--png", default="", help="Пусто → <output-dir>/quality_vs_tested_matrices.png")
    parser.add_argument("--plot-only", action="store_true", help="Не считать заново, а только построить график из CSV.")
    parser.add_argument("--w-negative", type=float, default=0.8)
    parser.add_argument("--w-glin", type=float, default=0.1)
    parser.add_argument("--w-coll", type=float, default=0.1)
    parser.add_argument("--population-size", type=int, default=DEFAULT_GA_POPULATION_SIZE)
    parser.add_argument("--ngen", type=int, default=110)
    parser.add_argument("--cxpb", type=float, default=DEFAULT_GA_CXPB)
    parser.add_argument("--mutpb", type=float, default=DEFAULT_GA_MUTPB)
    parser.add_argument("--indpb", type=float, default=DEFAULT_GA_INDPB)
    parser.add_argument("--tournsize", type=int, default=DEFAULT_GA_TOURNSIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_GA_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    csv_path = resolve_path(args.csv, PROJECT_ROOT) if args.csv else output_dir / "quality_vs_tested_matrices.csv"
    png_path = resolve_path(args.png, PROJECT_ROOT) if args.png else output_dir / "quality_vs_tested_matrices.png"

    if args.plot_only:
        rows = read_csv_rows(csv_path)
        plot_from_rows(rows, png_path)
        print(f"График записан: {png_path}")
        return

    las_path = resolve_path(args.las, PROJECT_ROOT)
    config_dir = resolve_path(args.config_dir, PROJECT_ROOT)

    data, _is_coll, _is_glin, _coll_prop, _glin_prop, _litho_raw = load_mkm_from_las(las_path, verbose=True)
    intervals = split_lithotype_intervals(data)

    a_min_coll = np.loadtxt(config_dir / "a_min_coll.in")
    a_max_coll = np.loadtxt(config_dir / "a_max_coll.in")
    a_k_coll = np.loadtxt(config_dir / "a_k_coll.in")
    a_min_glin = np.loadtxt(config_dir / "a_min_glin.in")
    a_max_glin = np.loadtxt(config_dir / "a_max_glin.in")
    a_k_glin = np.loadtxt(config_dir / "a_k_glin.in")

    for arr, name in (
        (a_min_coll, "A_min_coll"),
        (a_max_coll, "A_max_coll"),
        (a_min_glin, "A_min_glin"),
        (a_max_glin, "A_max_glin"),
    ):
        validate_matrix_shape(arr, name)
    validate_k_shape(a_k_coll, "A_k_coll")
    validate_k_shape(a_k_glin, "A_k_glin")

    rows: list[dict[str, object]] = []
    subdivisions = [int(item.strip()) for item in args.subdivisions.split(",") if item.strip()]

    for subdivision_count in subdivisions:
        bf_summary = run_interval_bruteforce(
            data=data,
            intervals=intervals,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_k_coll=build_subdivision_k_matrix(a_k_coll, subdivision_count),
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            a_k_glin=build_subdivision_k_matrix(a_k_glin, subdivision_count),
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            max_iterations=None,
            verbose=False,
        )
        rows.append(
            {
                "label": f"bf_{subdivision_count}",
                "method_family": "bf",
                "subdivisions": subdivision_count,
                "tested_matrices": bf_summary.total_evals,
                "quality_score": bf_summary.quality_score,
                "negative_share": bf_summary.negative_share,
                "glin_bad_share": bf_summary.glin_bad_share,
                "coll_bad_share": bf_summary.coll_bad_share,
                "time_sec": bf_summary.total_time_sec,
                "intervals": len(bf_summary.interval_results),
                "invalid_count": bf_summary.total_invalid_count,
                "generations": 0,
            }
        )
        print(
            f"bf_{subdivision_count}: Q={bf_summary.quality_score:.6f}, "
            f"tested={bf_summary.total_evals}, time={bf_summary.total_time_sec:.2f}s"
        )

    ga_params = GAParams(
        population_size=args.population_size,
        ngen=args.ngen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        indpb=args.indpb,
        tournsize=args.tournsize,
        patience=args.patience,
        min_delta=args.min_delta,
        n_jobs=1,
        seed=args.seed,
    )
    ga_summary = run_interval_ga(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll,
        a_max_coll=a_max_coll,
        a_min_glin=a_min_glin,
        a_max_glin=a_max_glin,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        ga_params=ga_params,
        verbose=False,
    )
    rows.append(
        {
            "label": "ga",
            "method_family": "ga",
            "subdivisions": "",
            "tested_matrices": ga_summary.total_evals,
            "quality_score": ga_summary.quality_score,
            "negative_share": ga_summary.negative_share,
            "glin_bad_share": ga_summary.glin_bad_share,
            "coll_bad_share": ga_summary.coll_bad_share,
            "time_sec": ga_summary.total_time_sec,
            "intervals": len(ga_summary.interval_results),
            "invalid_count": ga_summary.total_invalid_count,
            "generations": ga_summary.total_generations,
        }
    )
    print(
        f"ga: Q={ga_summary.quality_score:.6f}, "
        f"tested={ga_summary.total_evals}, time={ga_summary.total_time_sec:.2f}s"
    )

    write_csv_rows(csv_path, rows)
    plot_from_rows(rows, png_path)
    print(f"CSV записан: {csv_path}")
    print(f"График записан: {png_path}")


if __name__ == "__main__":
    main()
