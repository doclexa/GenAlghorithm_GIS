#!/usr/bin/env python3
"""Несколько прогонов интервального BF с разным coarse-factor: графики МКМ + сводка метрик/времени/evals."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT as MKM_PROJECT_ROOT,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    scale_mkm_model_for_metrics,
    split_lithotype_intervals,
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_interval_optimizer import (  # noqa: E402
    coarsen_k_matrix,
    run_interval_bruteforce,
)


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def coarse_label(f: float) -> str:
    if float(f).is_integer():
        return str(int(f))
    return str(f).replace(".", "_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--las", default=DEFAULT_LAS_RELPATH)
    p.add_argument("--config-dir", default="config")
    p.add_argument(
        "--coarse-factors",
        default="2,3,4,6,10",
        help="Список coarse-factor через запятую (>1).",
    )
    p.add_argument("--w-negative", type=float, default=0.8)
    p.add_argument("--w-glin", type=float, default=0.1)
    p.add_argument("--w-coll", type=float, default=0.1)
    p.add_argument(
        "--output-dir",
        default="outputs/experiments/bf_coarse_sweep",
        help="Каталог для CSV и сводного графика.",
    )
    p.add_argument(
        "--plots-dir",
        default="outputs/plots",
        help="Каталог для отдельных PNG МКМ по каждому factor.",
    )
    args = p.parse_args()

    out_dir = resolve_path(args.output_dir, MKM_PROJECT_ROOT)
    plots_dir = resolve_path(args.plots_dir, MKM_PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    factors = parse_float_list(args.coarse_factors)
    for f in factors:
        if f <= 1:
            raise ValueError(f"coarse-factor должен быть > 1, получено: {f}")

    las_path = resolve_path(args.las, MKM_PROJECT_ROOT)
    config_dir = resolve_path(args.config_dir, MKM_PROJECT_ROOT)

    data, _ic, _ig, _cp, _gp, litho_raw = load_mkm_from_las(las_path, verbose=True)
    intervals = split_lithotype_intervals(data)
    stem = las_path.stem

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

    print("=== Сверка BF по coarse-factor ===")
    print(f"Интервалов: {len(intervals)} | Q = {args.w_negative}*neg + {args.w_glin}*glin_bad + {args.w_coll}*coll_bad\n")

    for factor in factors:
        a_k_c = coarsen_k_matrix(a_k_coll, factor)
        a_k_g = coarsen_k_matrix(a_k_glin, factor)
        print(f"--- coarse_factor={factor:g} ---")
        summary = run_interval_bruteforce(
            data=data,
            intervals=intervals,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_k_coll=a_k_c,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            a_k_glin=a_k_g,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            max_iterations=None,
            verbose=False,
        )
        plot_path = plots_dir / f"{stem}_mkm_bf_coarse_{coarse_label(factor)}.png"
        mkm_plot = scale_mkm_model_for_metrics(summary.mkm_model)
        save_mkm_plot(
            mkm_plot,
            plot_path,
            litho_raw=litho_raw,
            litho_mnem="LITO",
            intervals=intervals,
        )
        row = {
            "coarse_factor": factor,
            "Q": summary.quality_score,
            "negative_share": summary.negative_share,
            "glin_bad_share": summary.glin_bad_share,
            "coll_bad_share": summary.coll_bad_share,
            "time_sec": summary.total_time_sec,
            "evals": summary.total_evals,
            "invalid_count": summary.total_invalid_count,
            "plot_png": str(plot_path.relative_to(MKM_PROJECT_ROOT)),
        }
        rows.append(row)
        print(
            f"  time={summary.total_time_sec:.3f}s | evals={summary.total_evals} | "
            f"invalid={summary.total_invalid_count}"
        )
        print(
            f"  Q={summary.quality_score:.8f} | neg={summary.negative_share:.6f} | "
            f"glin_bad={summary.glin_bad_share:.6f} | coll_bad={summary.coll_bad_share:.6f}"
        )
        print(f"  график: {plot_path}\n")

    csv_path = out_dir / "bf_coarse_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {csv_path}")

    xs = [float(r["coarse_factor"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_q, ax_t, ax_e, ax_neg = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_q.plot(xs, [r["Q"] for r in rows], "o-", color="#2980b9", linewidth=2)
    ax_q.set_xlabel("coarse_factor")
    ax_q.set_ylabel("Q (меньше лучше)")
    ax_q.set_title("Качество Q")
    ax_q.grid(True, alpha=0.3)

    ax_t.plot(xs, [r["time_sec"] for r in rows], "s-", color="#27ae60", linewidth=2)
    ax_t.set_xlabel("coarse_factor")
    ax_t.set_ylabel("Время, с")
    ax_t.set_title("Скорость перебора")
    ax_t.grid(True, alpha=0.3)

    ax_e.plot(xs, [r["evals"] for r in rows], "d-", color="#8e44ad", linewidth=2)
    ax_e.set_xlabel("coarse_factor")
    ax_e.set_ylabel("Число опробованных матриц")
    ax_e.set_title("Объём перебора (evals)")
    ax_e.grid(True, alpha=0.3)

    ax_neg.plot(xs, [r["negative_share"] for r in rows], "^-", color="#e67e22", linewidth=2)
    ax_neg.set_xlabel("coarse_factor")
    ax_neg.set_ylabel("negative_share")
    ax_neg.set_title("Доля отрицательных (после рескейла)")
    ax_neg.grid(True, alpha=0.3)

    fig.suptitle(f"BF: влияние coarse_factor ({stem})", fontsize=14)
    fig.tight_layout()
    summary_png = out_dir / "bf_coarse_sweep_dashboard.png"
    fig.savefig(summary_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Сводный график: {summary_png}")


if __name__ == "__main__":
    main()
