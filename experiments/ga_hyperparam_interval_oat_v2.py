#!/usr/bin/env python3
"""
v2: OAT (one-at-a-time) sweep гиперпараметров GA по сетке с фиксированным шагом.

На каждый узел сетки одного гиперпараметра остальные фиксируются на «стандартных» значениях
(как в mkm_run_ga по умолчанию). По каждому интервалу скважины запускается отдельный GA;
распределение local_score по интервалам задаёт один «ящик» boxplot для этого узла.

Артефакты для каждого гиперпараметра P:
  - oat_v2_{P}_detail.csv      — сырая сетка: интервал × значение
  - oat_v2_{P}_boxplot_summary.csv — сводка для ручной правки и перерисовки без пересчёта

How to run:
  python -u experiments/ga_hyperparam_interval_oat_v2.py --las data/las/621_1700_1780.las \\
    --output-dir outputs/experiments/ga_oat_interval_v2_skv621

Только отрисовка из detail-CSV (если уже были прогоны):
  python experiments/ga_hyperparam_interval_oat_v2.py --render-only --output-dir outputs/experiments/ga_oat_interval_v2_skv621

Только из summary (после ручного редактирования чисел):
  python experiments/ga_hyperparam_interval_oat_v2.py --render-only --from-summary \\
    --output-dir outputs/experiments/ga_oat_interval_v2_skv621
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

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
from mkm_ga_engine import (  # noqa: E402
    DEFAULT_GA_CXPB,
    DEFAULT_GA_INDPB,
    DEFAULT_GA_MUTPB,
    DEFAULT_GA_PATIENCE,
    DEFAULT_GA_POPULATION_SIZE,
    DEFAULT_GA_TOURNSIZE,
    GAParams,
)
from mkm_interval_optimizer import run_single_interval_ga  # noqa: E402

# Имена файлов
def detail_path(out_dir: Path, param: str) -> Path:
    return out_dir / f"oat_v2_{param}_detail.csv"


def summary_path(out_dir: Path, param: str) -> Path:
    return out_dir / f"oat_v2_{param}_boxplot_summary.csv"


def plot_path(out_dir: Path, param: str) -> Path:
    return out_dir / f"oat_v2_{param}_boxplot.png"


SUMMARY_FIELDNAMES = [
    "hyperparameter",
    "hyperparameter_value",
    "n_intervals",
    "mean_local_score",
    "std_local_score",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "whisker_low",
    "whisker_high",
    "mean_evals",
    "mean_elapsed_sec",
]

DETAIL_FIELDNAMES = [
    "hyperparameter",
    "hyperparameter_value",
    "interval_id",
    "lithotype",
    "depth_start",
    "depth_end",
    "sample_count",
    "local_score",
    "evals",
    "elapsed_sec",
    "generations_ran",
    "seed",
]


def _int_grid(lo: int, hi: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("step должен быть > 0.")
    cur = lo
    out: list[int] = []
    while cur <= hi:
        out.append(int(cur))
        cur += step
    if hi not in out:
        out.append(int(hi))
    return sorted(set(out))




def default_grids(args: argparse.Namespace) -> dict[str, list[float | int]]:
    """Широкие диапазоны с ровными шагами (редактируются CLI)."""
    return {
        "population_size": _int_grid(200, 1000, int(args.step_pop)),
        "ngen": _int_grid(40, 400, int(args.step_ngen)),
        "cxpb": [
            round(x, 8)
            for x in np.arange(float(args.cxpb_lo), float(args.cxpb_hi) + args.step_cxpb * 0.5, float(args.step_cxpb))
        ],
        "mutpb": [
            round(x, 8)
            for x in np.arange(float(args.mutpb_lo), float(args.mutpb_hi) + args.step_mutpb * 0.5, float(args.step_mutpb))
        ],
        "indpb": [
            round(x, 8)
            for x in np.arange(float(args.indpb_lo), float(args.indpb_hi) + args.step_indpb * 0.5, float(args.step_indpb))
        ],
        "tournsize": _int_grid(int(args.tour_lo), int(args.tour_hi), int(args.step_tour)),
        "patience": _int_grid(int(args.patience_lo), int(args.patience_hi), int(args.step_patience)),
        "min_delta": [float(v) for v in args.min_delta_grid.split(",") if str(v).strip()],
    }


def baseline_ga(args: argparse.Namespace) -> GAParams:
    return GAParams(
        population_size=int(args.baseline_pop),
        ngen=int(args.baseline_ngen),
        cxpb=float(args.baseline_cxpb),
        mutpb=float(args.baseline_mutpb),
        indpb=float(args.baseline_indpb),
        tournsize=int(args.baseline_tournsize),
        patience=int(args.baseline_patience),
        min_delta=float(args.baseline_min_delta),
        n_jobs=max(1, int(args.n_jobs)),
        seed=int(args.baseline_seed) & 0x7FFFFFFF,
    )


def coerce_param(param: str, value: float | int) -> float | int:
    if param in ("population_size", "ngen", "tournsize", "patience"):
        return int(round(float(value)))
    return float(value)


def with_hyperparam(base: GAParams, param: str, value: float | int) -> GAParams:
    coerced = coerce_param(param, value)
    return replace(base, **{param: coerced})


def value_label(param: str, value: Any) -> str:
    if param == "min_delta":
        return f"{float(value):.2e}".replace("+", "")
    fv = float(value)
    if param in ("population_size", "ngen", "tournsize", "patience"):
        return str(int(round(fv)))
    s = f"{fv:.10g}"
    return s


def _sort_tick_label(label: str) -> float:
    try:
        return float(str(label).strip())
    except ValueError:
        return 0.0


def compute_box_summaries(samples: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
    """mean, std, min, q1, med, q3, max, whisker_low, whisker_high (Tukey без выбросов в флайерах)."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    lo, hi = float(x[0]), float(x[-1])
    q1, med, q3 = tuple(np.percentile(x, [25.0, 50.0, 75.0]).tolist())
    iqr = q3 - q1
    wlo = max(lo, float(q1 - 1.5 * iqr))
    whi = min(hi, float(q3 + 1.5 * iqr))
    return mean, std, lo, float(q1), float(med), float(q3), hi, float(wlo), float(whi)


def summarize_detail_rows(rows: list[dict[str, str]], param_name: str) -> list[dict[str, Any]]:
    by_val: dict[str, list[float]] = {}
    by_val_e = dict[str, list[float]]()
    by_val_t = dict[str, list[float]]()
    for row in rows:
        if row.get("hyperparameter") != param_name:
            continue
        vk = row["hyperparameter_value"].strip()
        by_val.setdefault(vk, []).append(float(row["local_score"]))
        by_val_e.setdefault(vk, []).append(float(row["evals"]))
        by_val_t.setdefault(vk, []).append(float(row["elapsed_sec"]))
    summaries: list[dict[str, Any]] = []
    for vk in sorted(by_val.keys(), key=_sort_tick_label):
        s = np.array(by_val[vk])
        mean, std, vmin, q1, med, q3, vmax, wl, wh = compute_box_summaries(s)
        summaries.append(
            {
                "hyperparameter": param_name,
                "hyperparameter_value": vk,
                "n_intervals": len(s),
                "mean_local_score": mean,
                "std_local_score": std,
                "min": vmin,
                "q1": q1,
                "median": med,
                "q3": q3,
                "max": vmax,
                "whisker_low": wl,
                "whisker_high": wh,
                "mean_evals": float(np.mean(by_val_e[vk])) if vk in by_val_e else "",
                "mean_elapsed_sec": float(np.mean(by_val_t[vk])) if vk in by_val_t else "",
            }
        )
    return summaries


def save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()
        for r in rows:
            out = {}
            for k in SUMMARY_FIELDNAMES:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.10g}"
                elif v == "":
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def plot_param_from_detail(out_dir: Path, param_name: str) -> None:
    path = detail_path(out_dir, param_name)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    by_val: dict[str, list[float]] = {}
    mean_n_by: dict[str, float] = {}
    mean_t_by: dict[str, float] = {}
    vk_order: list[str] = []

    tmp: dict[str, list[tuple[list[float], list[float]]]] = {}
    for row in rows:
        if row["hyperparameter"] != param_name:
            continue
        vk = row["hyperparameter_value"].strip()
        if vk not in tmp:
            tmp[vk] = []
            vk_order.append(vk)
        by_val.setdefault(vk, []).append(float(row["local_score"]))
    # preserve first-seen vk order sorted numerically preferred
    def sort_key_lab(lab: str) -> tuple:
        try:
            return (0, float(lab))
        except ValueError:
            try:
                return (0, int(lab))
            except ValueError:
                return (1, lab)

    vk_order_unique = sorted(vk_order, key=sort_key_lab)
    for vk in vk_order_unique:
        ev = [float(r["evals"]) for r in rows if r["hyperparameter_value"].strip() == vk]
        tv = [float(r["elapsed_sec"]) for r in rows if r["hyperparameter_value"].strip() == vk]
        mean_n_by[vk] = float(np.mean(ev)) if ev else 0.0
        mean_t_by[vk] = float(np.mean(tv)) if tv else 0.0

    groups = [by_val[vk] for vk in vk_order_unique if vk in by_val]

    positions = np.arange(1, len(groups) + 1)
    fig_w = max(9.0, 0.72 * len(groups) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    try:
        ax.boxplot(groups, positions=positions, tick_labels=vk_order_unique)
    except TypeError:
        ax.boxplot(groups, positions=positions, labels=vk_order_unique)

    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-12)
    ax.set_ylim(ylo - 0.42 * span, yhi + 0.06 * span)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    y_floor = ax.get_ylim()[0]
    for i, vk in enumerate(vk_order_unique):
        xc = float(positions[i])
        mn = mean_n_by.get(vk, 0.0)
        mt = mean_t_by.get(vk, 0.0)
        ax.annotate(
            f"mean N={mn:,.0f}\nmean t={mt:.2f}s",
            xy=(xc, y_floor),
            xycoords=("data", "data"),
            xytext=(0, -56),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
        )

    ax.set_ylabel("local_score по интервалу (меньше лучше, как GA fitness)")
    ax.set_xlabel(param_name)
    ax.set_title(f"OAT по сетке: {param_name} (ящик — распределение по интервалам)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.42)
    out_png = plot_path(out_dir, param_name)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out_png}", flush=True)


def plot_param_from_summary(out_dir: Path, param_name: str) -> None:
    """bxp только по числам из summary CSV (можно редактировать вручную)."""
    sp = summary_path(out_dir, param_name)
    if not sp.is_file():
        return
    with sp.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return

    ticks = []
    stats = []
    for row in reader:
        ticks.append(row["hyperparameter_value"])
        stats.append(
            {
                "med": float(row["median"]),
                "q1": float(row["q1"]),
                "q3": float(row["q3"]),
                "whislo": float(row["whisker_low"]),
                "whishi": float(row["whisker_high"]),
                "fliers": [],
                "mean": float(row["mean_local_score"]),
                "label": row["hyperparameter_value"],
            }
        )

    positions = np.arange(1, len(stats) + 1)
    fig_w = max(9.0, 0.72 * len(stats) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    ax.bxp(stats, positions=positions, showfliers=False, showmeans=True)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(ticks, rotation=30, ha="right")
    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-12)
    ax.set_ylim(ylo - 0.42 * span, yhi + 0.06 * span)
    y_floor = ax.get_ylim()[0]
    for i, row in enumerate(reader):
        xc = float(positions[i])
        mn = row.get("mean_evals", "")
        mt = row.get("mean_elapsed_sec", "")
        try:
            mnf = float(mn)
            txt_mn = f"{mnf:,.0f}"
        except (TypeError, ValueError):
            txt_mn = str(mn)
        try:
            mtf = float(mt)
            txt_mt = f"{mtf:.2f}s"
        except (TypeError, ValueError):
            txt_mt = str(mt)
        ax.annotate(
            f"mean N={txt_mn}\nmean t={txt_mt}",
            xy=(xc, y_floor),
            xycoords=("data", "data"),
            xytext=(0, -56),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
        )

    ax.set_ylabel("local_score (из summary)")
    ax.set_xlabel(param_name)
    ax.set_title(f"{param_name} — boxplot из summary CSV (можно было править вручную)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.42)
    out_png = plot_path(out_dir, param_name)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"График (summary): {out_png}", flush=True)


def run_sweep_parameter(
    *,
    param_name: str,
    values: Iterable[float | int],
    intervals: list,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    base_ga: GAParams,
    seed_base: int,
    counter_start: int,
    w_negative: float,
    w_glin: float,
    w_coll: float,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    run_idx = counter_start
    vals = sorted(set(values), key=float)
    for v in vals:
        ga_tpl = replace(base_ga, seed=(seed_base + run_idx * 7937) & 0x7FFFFFFF)
        gp = with_hyperparam(ga_tpl, param_name, v)
        lbl = value_label(param_name, v)
        for iv in intervals:
            run_idx += 1
            gps = replace(gp, seed=(seed_base + run_idx * 104729) & 0x7FFFFFFF)
            started = time.perf_counter()
            res = run_single_interval_ga(
                iv,
                a_min_coll=a_min_coll,
                a_max_coll=a_max_coll,
                a_min_glin=a_min_glin,
                a_max_glin=a_max_glin,
                w_negative=w_negative,
                w_glin=w_glin,
                w_coll=w_coll,
                ga_params=gps,
                verbose=False,
            )
            dt = time.perf_counter() - started
            rows.append(
                {
                    "hyperparameter": param_name,
                    "hyperparameter_value": lbl,
                    "interval_id": res.interval_id,
                    "lithotype": res.lithotype,
                    "depth_start": f"{res.depth_start:.6g}",
                    "depth_end": f"{res.depth_end:.6g}",
                    "sample_count": res.sample_count,
                    "local_score": f"{res.local_score:.10g}",
                    "evals": res.evals,
                    "elapsed_sec": f"{res.elapsed_sec:.6g}",
                    "generations_ran": res.generations_ran,
                    "seed": gps.seed if gps.seed is not None else "",
                    "_wall_sec": dt,
                }
            )
            print(
                f"   [ {param_name}={lbl} ] interval={res.interval_id} Qloc={res.local_score:.8f} "
                f"N={res.evals} gen={res.generations_ran} t={dt:.2f}s",
                flush=True,
            )
    return rows, run_idx


def save_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in DETAIL_FIELDNAMES if k != "_wall_sec"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ga_hyperparam_interval_oat_v2: OAT + сетка + боксы по интервалам.")
    p.add_argument("--las", default=DEFAULT_LAS_RELPATH)
    p.add_argument("--config-dir", default="config")
    p.add_argument("--output-dir", default="outputs/experiments/ga_oat_interval_v2_skv621")
    p.add_argument("--w-negative", type=float, default=0.8)
    p.add_argument("--w-glin", type=float, default=0.1)
    p.add_argument("--w-coll", type=float, default=0.1)
    p.add_argument("--n-jobs", type=int, default=1)
    # сетки
    p.add_argument("--step-pop", type=int, default=80, dest="step_pop")
    p.add_argument("--step-ngen", type=int, default=50, dest="step_ngen")
    p.add_argument("--cxpb-lo", type=float, default=0.4)
    p.add_argument("--cxpb-hi", type=float, default=0.85)
    p.add_argument("--step-cxpb", type=float, default=0.05, dest="step_cxpb")
    p.add_argument("--mutpb-lo", type=float, default=0.1)
    p.add_argument("--mutpb-hi", type=float, default=1.0)
    p.add_argument("--step-mutpb", type=float, default=0.05, dest="step_mutpb")
    p.add_argument("--indpb-lo", type=float, default=0.02)
    p.add_argument("--indpb-hi", type=float, default=0.55)
    p.add_argument("--step-indpb", type=float, default=0.03, dest="step_indpb")
    p.add_argument("--tour-lo", type=int, default=2)
    p.add_argument("--tour-hi", type=int, default=18)
    p.add_argument("--step-tour", type=int, default=2, dest="step_tour")
    p.add_argument("--patience-lo", type=int, default=0)
    p.add_argument("--patience-hi", type=int, default=150)
    p.add_argument("--step-patience", type=int, default=15, dest="step_patience")
    p.add_argument(
        "--min-delta-grid",
        default="1e-8,5e-8,1e-7,5e-7,2e-6,1e-6",
        dest="min_delta_grid",
        help="Список min_delta через запятую.",
    )

    # baseline = общие дефолты GA (см. mkm_ga_engine.DEFAULT_GA_* и mkm_run_ga)
    p.add_argument("--baseline-pop", type=int, default=DEFAULT_GA_POPULATION_SIZE)
    p.add_argument("--baseline-ngen", type=int, default=110)
    p.add_argument("--baseline-cxpb", type=float, default=DEFAULT_GA_CXPB)
    p.add_argument("--baseline-mutpb", type=float, default=DEFAULT_GA_MUTPB)
    p.add_argument("--baseline-indpb", type=float, default=DEFAULT_GA_INDPB)
    p.add_argument("--baseline-tournsize", type=int, default=DEFAULT_GA_TOURNSIZE)
    p.add_argument("--baseline-patience", type=int, default=DEFAULT_GA_PATIENCE)
    p.add_argument("--baseline-min-delta", type=float, default=1e-7)
    p.add_argument("--baseline-seed", type=int, default=2026)

    p.add_argument("--seed-base", type=int, default=10_013, dest="seed_base")
    p.add_argument("--render-only", action="store_true", help="Не считать, только построить графики из CSV.")
    p.add_argument("--from-summary", action="store_true", help="С render-only — читать только *_boxplot_summary.csv.")
    p.add_argument(
        "--only-params",
        default="",
        help="Подмножество имён параметров через запятую, напр.: cxpb,mutpb",
    )
    p.add_argument("--recalc-summary", action="store_true", help="С render-only — перезаписать summary из detail.")
    return p.parse_args()


def discover_params_detail(out_dir: Path) -> list[str]:
    found: list[str] = []
    for pth in sorted(out_dir.glob("oat_v2_*_detail.csv")):
        name = pth.stem
        suf = "_detail"
        if name.endswith(suf):
            mid = name[len("oat_v2_") : -len(suf)]
            if mid:
                found.append(mid)
    return found


def discover_params_summary(out_dir: Path) -> list[str]:
    found: list[str] = []
    suf = "_boxplot_summary"
    pre = "oat_v2_"
    for pth in sorted(out_dir.glob("oat_v2_*_boxplot_summary.csv")):
        stem = pth.stem
        if stem.endswith(suf):
            mid = stem[len(pre) : -len(suf)]
            if mid:
                found.append(mid)
    return found


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    out_dir = resolve_path(args.output_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = default_grids(args)
    order = ["population_size", "ngen", "cxpb", "mutpb", "indpb", "tournsize", "patience", "min_delta"]
    if args.only_params.strip():
        allow = {s.strip() for s in args.only_params.split(",") if s.strip()}
        order = [k for k in order if k in allow]

    if args.render_only:
        if args.from_summary:
            params = discover_params_summary(out_dir)
        else:
            params = discover_params_detail(out_dir)
        if args.only_params.strip():
            allow = {s.strip() for s in args.only_params.split(",") if s.strip()}
            params = [p for p in params if p in allow]
        if not params:
            print("Нет подходящих CSV (detail или summary) в каталоге.", flush=True)
        for pname in params:
            detail_p = detail_path(out_dir, pname)
            summary_p = summary_path(out_dir, pname)
            if not args.from_summary:
                if args.recalc_summary and detail_p.is_file():
                    with detail_p.open(newline="", encoding="utf-8") as f:
                        drows = list(csv.DictReader(f))
                    sums = summarize_detail_rows(drows, pname)
                    save_summary_csv(summary_p, sums)
                    print(f"Перезапись summary: {summary_p}", flush=True)
                if detail_p.is_file():
                    plot_param_from_detail(out_dir, pname)
                    if args.recalc_summary or not summary_p.is_file():
                        with detail_p.open(newline="", encoding="utf-8") as f:
                            drows = list(csv.DictReader(f))
                        sums = summarize_detail_rows(drows, pname)
                        save_summary_csv(summary_p, sums)
                else:
                    print(f"Пропуск {pname}: нет detail CSV", flush=True)
            else:
                if summary_p.is_file():
                    plot_param_from_summary(out_dir, pname)
                else:
                    print(f"Пропуск {pname}: нет summary CSV", flush=True)
        print("Готово (render-only).", flush=True)
        return

    cfg = resolve_path(args.config_dir, project_root)
    las_path = resolve_path(args.las, project_root)
    data, _ic, _ig, _cp, _gp, _lr = load_mkm_from_las(las_path, verbose=True)
    intervals = split_lithotype_intervals(data)

    a_min_coll = np.loadtxt(cfg / "a_min_coll.in")
    a_max_coll = np.loadtxt(cfg / "a_max_coll.in")
    a_min_glin = np.loadtxt(cfg / "a_min_glin.in")
    a_max_glin = np.loadtxt(cfg / "a_max_glin.in")
    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    base = baseline_ga(args)
    w_neg, w_gl, w_cl = args.w_negative, args.w_glin, args.w_coll

    total_cells = sum(len(grids[p]) for p in order if p in grids)
    n_iv = len(intervals)
    print(
        f"Интервалов: {n_iv}. Оценка полных запусков GA: {total_cells * n_iv} "
        f"({total_cells} точек сетки x {n_iv} интервалов).",
        flush=True,
    )

    run_counter = 0
    t_all = time.perf_counter()
    for pname in order:
        if pname not in grids:
            continue
        vals = grids[pname]
        print(f"\n=== Sweep {pname}: {len(vals)} значений сетки ===", flush=True)
        rows_part, run_counter = run_sweep_parameter(
            param_name=pname,
            values=vals,
            intervals=intervals,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            base_ga=base,
            seed_base=int(args.seed_base),
            counter_start=run_counter,
            w_negative=w_neg,
            w_glin=w_gl,
            w_coll=w_cl,
        )
        save_detail_csv(detail_path(out_dir, pname), rows_part)
        sums = summarize_detail_rows(
            [
                {
                    "hyperparameter": str(r["hyperparameter"]),
                    "hyperparameter_value": str(r["hyperparameter_value"]),
                    "local_score": str(r["local_score"]),
                    "evals": str(r["evals"]),
                    "elapsed_sec": str(r["elapsed_sec"]),
                }
                for r in rows_part
            ],
            pname,
        )
        save_summary_csv(summary_path(out_dir, pname), sums)
        plot_param_from_detail(out_dir, pname)
        print(f"CSV: {detail_path(out_dir, pname)}\nCSV: {summary_path(out_dir, pname)}", flush=True)

    print(f"\nВсего времени sweep: {time.perf_counter() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
