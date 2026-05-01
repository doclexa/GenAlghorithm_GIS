#!/usr/bin/env python3
"""
Глобальное исследование гиперпараметров интервального GA (Random / LHS).

По умолчанию: data/las/621_1700_1780.las.

Графики: только boxplot Q по квантильным бинам каждого гиперпараметра (кроме min_delta);
под каждым бином подписаны среднее число оценённых матриц и среднее время.

How to run locally:
  python experiments/ga_hyperparam_global_study.py --las data/las/621_1700_1780.las \\
    --output-dir outputs/experiments/ga_global_skv621

Только перерисовка из CSV:

  python experiments/ga_hyperparam_global_study.py --plot-only \\
    --output-dir outputs/experiments/ga_global_skv621

How to run in Docker: аналогично со смонтированным репозиторием.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    load_mkm_from_las,
    resolve_path,
    split_lithotype_intervals,
    validate_matrix_shape,
)
from mkm_ga_engine import GAParams  # noqa: E402
from mkm_interval_optimizer import run_interval_ga  # noqa: E402

CSV_FIELDNAMES: list[str] = [
    "trial_id",
    "seed",
    "sampler",
    "population_size",
    "ngen",
    "cxpb",
    "mutpb",
    "indpb",
    "tournsize",
    "patience",
    "min_delta",
    "n_jobs",
    "n_intervals",
    "final_Q",
    "total_time_sec",
    "total_tested_matrices",
    "negative_share",
    "glin_bad_share",
    "coll_bad_share",
]

_LHS_DIMS = 8
# Диапазон min_delta не менять (ось 7 LHS)
_LOG_MIN_DELTA = (1e-8, 1e-6)

# Поля для boxplot по бинам (min_delta исключён)
BOX_PLOT_KEYS: list[str] = [
    "population_size",
    "ngen",
    "cxpb",
    "mutpb",
    "indpb",
    "tournsize",
    "patience",
]

# Поля числового разбора CSV (учёт столбца min_delta в данных без графика)
HP_PARSE_KEYS = BOX_PLOT_KEYS + ["min_delta"]


def _map_unit_to_params(
    row_u: np.ndarray,
    n_jobs: int,
    base_seed: int,
    trial_id: int,
) -> GAParams:
    """row_u: shape (8,) в [0,1); cxpb как раньше; min_delta только по прежнему лог диапазону."""
    pop = int(np.clip(np.round(200.0 + row_u[0] * (1000.0 - 200.0)), 200, 1000))
    ngen = int(np.clip(np.round(40.0 + row_u[1] * (400.0 - 40.0)), 40, 400))

    cxpb = 0.4 + row_u[2] * (0.85 - 0.4)
    mutpb = np.clip(0.1 + row_u[3] * (1.0 - 0.1), 0.1, 1.0)
    indpb = np.clip(0.02 + row_u[4] * (0.55 - 0.02), 0.02, 0.55)

    tour = int(np.clip(np.round(2.0 + row_u[5] * (18.0 - 2.0)), 2, 18))
    pat = int(np.clip(np.round(row_u[6] * 150.0), 0, 150))

    log_lo, log_hi = np.log10(_LOG_MIN_DELTA[0]), np.log10(_LOG_MIN_DELTA[1])
    min_d = 10.0 ** (log_lo + row_u[7] * (log_hi - log_lo))
    seed = int(base_seed + trial_id) & 0x7FFFFFFF
    return GAParams(
        population_size=int(pop),
        ngen=int(ngen),
        cxpb=float(cxpb),
        mutpb=float(mutpb),
        indpb=float(indpb),
        tournsize=int(tour),
        patience=int(pat),
        min_delta=float(min_d),
        n_jobs=n_jobs,
        seed=seed,
    )


def latin_hypercube_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n_samples, n_dim), dtype=np.float64)
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        u = rng.random(n_samples)
        out[:, j] = (perm + u) / n_samples
    return out


def random_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n_samples, n_dim), dtype=np.float64)


def row_dict_from_ga_params(
    trial_id: int,
    ga: GAParams,
    *,
    sampler: str,
    n_intervals: int,
    summary,
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "seed": ga.seed if ga.seed is not None else "",
        "sampler": sampler,
        "population_size": ga.population_size,
        "ngen": ga.ngen,
        "cxpb": f"{ga.cxpb:.10f}",
        "mutpb": f"{ga.mutpb:.10f}",
        "indpb": f"{ga.indpb:.10f}",
        "tournsize": ga.tournsize,
        "patience": ga.patience,
        "min_delta": f"{ga.min_delta:.10e}",
        "n_jobs": ga.n_jobs,
        "n_intervals": n_intervals,
        "final_Q": f"{summary.quality_score:.10f}",
        "total_time_sec": f"{summary.total_time_sec:.6f}",
        "total_tested_matrices": summary.total_evals,
        "negative_share": f"{summary.negative_share:.10f}",
        "glin_bad_share": f"{summary.glin_bad_share:.10f}",
        "coll_bad_share": f"{summary.coll_bad_share:.10f}",
    }


def read_trials_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_row_numeric(row: dict[str, str]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    keys = HP_PARSE_KEYS + ["final_Q", "total_time_sec", "total_tested_matrices"]
    for k in keys:
        if k not in row or row[k] is None or str(row[k]).strip() == "":
            continue
        v = str(row[k]).strip()
        if k in ("population_size", "ngen", "tournsize", "patience", "total_tested_matrices"):
            out[k] = int(float(v))
        else:
            out[k] = float(v)
    return out




def plot_boxplots_only(rows: list[dict[str, str]], out_dir: Path, n_bins: int) -> None:
    """Только boxplot Q по квантилям каждого HP; под бинами — средние N и время."""
    parsed = [_parse_row_numeric(r) for r in rows if r.get("final_Q")]
    if len(parsed) < max(10, n_bins + 3):
        return
    nb = max(3, min(n_bins, len(parsed) // 3))

    for key in BOX_PLOT_KEYS:
        good = [
            p
            for p in parsed
            if key in p
            and "final_Q" in p
            and "total_time_sec" in p
            and "total_tested_matrices" in p
        ]
        if len(good) < nb:
            continue
        x_vals = np.array([float(p[key]) for p in good], dtype=np.float64)
        q_vals = np.array([float(p["final_Q"]) for p in good], dtype=np.float64)
        t_vals = np.array([float(p["total_time_sec"]) for p in good], dtype=np.float64)
        n_vals = np.array([float(p["total_tested_matrices"]) for p in good], dtype=np.float64)

        q_edges = np.quantile(x_vals, np.linspace(0, 1, nb + 1))
        q_edges[0] = float(x_vals.min())
        q_edges[-1] = float(x_vals.max())
        bin_id = np.digitize(x_vals, q_edges[1:-1], right=False)
        bin_id = np.clip(bin_id, 0, nb - 1)

        groups: list[np.ndarray] = []
        tick_labels: list[str] = []
        mean_n: list[float] = []
        mean_t: list[float] = []
        counts: list[int] = []

        for b in range(nb):
            mask = bin_id == b
            if np.sum(mask) == 0:
                continue
            groups.append(q_vals[mask])
            lo, hi = q_edges[b], q_edges[b + 1]
            if key == "population_size":
                tick_labels.append(f"{int(lo)}–{int(hi)}")
            elif key in ("cxpb", "mutpb", "indpb"):
                tick_labels.append(f"{lo:.3g}–{hi:.3g}")
            else:
                tick_labels.append(f"{int(round(lo))}–{int(round(hi))}")
            mean_n.append(float(np.mean(n_vals[mask])))
            mean_t.append(float(np.mean(t_vals[mask])))
            counts.append(int(np.sum(mask)))

        if len(groups) < 2:
            continue

        fig_h = max(6.0, 1.35 * len(groups) * 0.55)
        fig, ax = plt.subplots(figsize=(11, fig_h))

        xpos = np.arange(1, len(groups) + 1, dtype=np.float64)
        try:
            ax.boxplot(groups, positions=list(xpos), tick_labels=tick_labels)
        except TypeError:
            ax.boxplot(groups, positions=list(xpos), labels=tick_labels)

        ax.set_ylabel("Q (меньше лучше)")
        ax.set_xlabel(key)
        ax.set_title(f"Распределение Q по квантильным бинам: {key} (n проб = {len(good)})")
        ax.grid(True, axis="y", alpha=0.3)

        ylo = float(min(g.min() for g in groups))
        yhi = float(max(g.max() for g in groups))
        span = max(yhi - ylo, 1e-12)
        ax.set_ylim(ylo - 0.32 * span, yhi + 0.06 * span)

        plt.setp(ax.get_xticklabels(), rotation=22, ha="right")

        for i in range(len(mean_n)):
            xc = float(xpos[i])
            ax.annotate(
                f"mean N={mean_n[i]:,.0f}\nmean t={mean_t[i]:.1f} s\n(n={counts[i]})",
                xy=(xc, ylo),
                xycoords=("data", "data"),
                xytext=(0, -48),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color="dimgray",
            )

        foot = (
            "Среднее по всем пробам этого графика: "
            f"N={float(np.mean(n_vals)):,.0f}, t={float(np.mean(t_vals)):.2f}s"
        )
        fig.subplots_adjust(bottom=0.38)
        fig.text(0.5, 0.02, foot, ha="center", fontsize=9)
        safe = str(key).replace(" ", "_")
        out_path = out_dir / f"ga_global_boxplot_Q_by_bins_{safe}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Глобальное исследование гиперпараметров интервального GA; выход — CSV и только boxplot-графики."
    )
    p.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las (по умолчанию data/las/621_1700_1780.las).",
    )
    p.add_argument("--config-dir", default="config")
    p.add_argument("--w-negative", type=float, default=0.8)
    p.add_argument("--w-glin", type=float, default=0.1)
    p.add_argument("--w-coll", type=float, default=0.1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument(
        "--sampler",
        choices=("random", "lhs"),
        default="lhs",
        help="Random или Latin Hypercube в [0,1]^8.",
    )
    p.add_argument("--trials", type=int, default=32, help="Число комбинаций гиперпараметров.")
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument(
        "--output-dir",
        default="outputs/experiments/ga_global_skv621",
        help="CSV и PNG (только ga_global_boxplot_*.png).",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Имя CSV (по умолчанию ga_global_trials.csv в --output-dir).",
    )
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--box-bins", type=int, default=5, help="Целевое число квантильных бинов.")
    p.add_argument("--project-root", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    out_dir = resolve_path(args.output_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv:
        csv_path = resolve_path(args.csv, project_root)
    else:
        csv_path = out_dir / "ga_global_trials.csv"

    if args.plot_only:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Нет CSV: {csv_path}")
        rows = read_trials_csv(csv_path)
        plot_boxplots_only(rows, out_dir, max(3, int(args.box_bins)))
        print(f"Boxplot-графики: {out_dir}", flush=True)
        return

    las_path = resolve_path(args.las, project_root)
    config_dir = resolve_path(args.config_dir, project_root)
    data, _ic, _ig, _cp, _gp, _lraw = load_mkm_from_las(las_path, verbose=True)
    intervals = split_lithotype_intervals(data)
    n_intervals = len(intervals)

    a_min_coll = np.loadtxt(config_dir / "a_min_coll.in")
    a_max_coll = np.loadtxt(config_dir / "a_max_coll.in")
    a_min_glin = np.loadtxt(config_dir / "a_min_glin.in")
    a_max_glin = np.loadtxt(config_dir / "a_max_glin.in")
    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    rng = np.random.default_rng(int(args.base_seed) % (2**32))
    n_trials = max(1, int(args.trials))
    if args.sampler == "lhs":
        u_batch = latin_hypercube_unit(n_trials, _LHS_DIMS, rng)
    else:
        u_batch = random_unit(n_trials, _LHS_DIMS, rng)

    n_jobs = max(1, int(args.n_jobs))
    rows_out: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for t in range(n_trials):
        ga = _map_unit_to_params(u_batch[t], n_jobs, int(args.base_seed), t)
        t_run = time.perf_counter()
        summary = run_interval_ga(
            data=data,
            intervals=intervals,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            ga_params=ga,
            verbose=False,
        )
        run_sec = time.perf_counter() - t_run
        row = row_dict_from_ga_params(
            t + 1,
            ga,
            sampler=args.sampler,
            n_intervals=n_intervals,
            summary=summary,
        )
        rows_out.append(row)
        print(
            f"Trial {t + 1}/{n_trials}  Q={summary.quality_score:.8f}  "
            f"N={summary.total_evals}  time={run_sec:.2f}s  seed={ga.seed}  "
            f"pop={ga.population_size} ngen={ga.ngen}",
            flush=True,
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})

    total = time.perf_counter() - t0
    print(f"Сводка: {n_trials} trial за {total:.1f} s -> {csv_path}", flush=True)

    str_rows = [{k: str(r.get(k, "")) for k in CSV_FIELDNAMES} for r in rows_out]
    plot_boxplots_only(str_rows, out_dir, max(3, int(args.box_bins)))
    print(f"Boxplot-графики: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
