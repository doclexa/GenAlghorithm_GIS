#!/usr/bin/env python3
"""
Сравнение bruteforce и GA на LAS (по умолчанию skv621.las): бюджет оценок, полный Q,
эксперимент устойчивости к сдвигу границ поиска.

Запуск из корня GenAlghorithm_GIS:
  python experiments/compare_bf_ga_skv621.py --all
  python experiments/compare_bf_ga_skv621.py --benchmark
  python experiments/compare_bf_ga_skv621.py --stability
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from math import prod
from pathlib import Path

import numpy as np

# корень проекта = родитель каталога experiments/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkm_bruteforce_engine import (  # noqa: E402
    brute_force_best_coll,
    brute_force_best_glin,
    build_value_grids,
)
from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    calc_mkm_model,
    calc_metrics_mkm,
    calc_quality_score,
    load_mkm_from_las,
    resolve_path,
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_ga_engine import GAParams, optimize_mkm_with_ga  # noqa: E402


def translate_bounds(
    a_min: np.ndarray,
    a_max: np.ndarray,
    frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Параллельный сдвиг бокса: a' = a + frac * (a_max - a_min) поэлементно."""
    span = np.asarray(a_max, dtype=float) - np.asarray(a_min, dtype=float)
    delta = frac * span
    return np.asarray(a_min, dtype=float) + delta, np.asarray(a_max, dtype=float) + delta


def run_ga_once(
    *,
    data: np.ndarray,
    is_coll: np.ndarray,
    is_glin: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    ga_params: GAParams,
) -> dict[str, float | int]:
    t0 = time.perf_counter()
    coll_r, glin_r, search_sec = optimize_mkm_with_ga(
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_min_coll=a_min_coll,
        a_max_coll=a_max_coll,
        a_min_glin=a_min_glin,
        a_max_glin=a_max_glin,
        w_negative=w_negative,
        w_glin=w_glin,
        w_coll=w_coll,
        ga_params=ga_params,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    mkm = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=coll_r.best_matrix,
        a_glin=glin_r.best_matrix,
    )
    neg_s, gb, cb = calc_metrics_mkm(mkm)
    q = calc_quality_score(neg_s, gb, cb, w_negative, w_glin, w_coll)
    nevals = int(coll_r.fitness_evals + glin_r.fitness_evals)
    return {
        "Q": float(q),
        "negative_share": float(neg_s),
        "glin_bad_share": float(gb),
        "coll_bad_share": float(cb),
        "time_sec": float(search_sec),
        "wall_sec": float(wall),
        "ga_fitness_evals": nevals,
        "coll_generations": int(coll_r.generations_ran),
        "glin_generations": int(glin_r.generations_ran),
    }


def run_bf_once(
    *,
    data: np.ndarray,
    is_coll: np.ndarray,
    is_glin: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_k_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    a_k_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    max_iter_coll: int | None,
    max_iter_glin: int | None,
) -> dict[str, float | int]:
    coll_ratio = len(coll_prop) / (len(coll_prop) + len(glin_prop))
    glin_ratio = len(glin_prop) / (len(coll_prop) + len(glin_prop))
    coll_grids = build_value_grids(a_min_coll, a_max_coll, a_k_coll)
    glin_grids = build_value_grids(a_min_glin, a_max_glin, a_k_glin)
    total_coll = prod(len(g) for g in coll_grids)
    total_glin = prod(len(g) for g in glin_grids)
    total_coll_e = min(total_coll, max_iter_coll) if max_iter_coll is not None else total_coll
    total_glin_e = min(total_glin, max_iter_glin) if max_iter_glin is not None else total_glin
    total_iters = total_coll_e + total_glin_e

    t0 = time.perf_counter()
    best_coll, _, _, _, coll_iters, _ = brute_force_best_coll(
        coll_prop=coll_prop,
        a_min=a_min_coll,
        a_max=a_max_coll,
        a_k=a_k_coll,
        neg_weight_scaled=w_negative * coll_ratio,
        coll_weight=w_coll,
        global_start_index=0,
        global_total=total_iters,
        max_iterations=max_iter_coll,
        verbose=False,
    )
    best_glin, _, _, _, glin_iters, _ = brute_force_best_glin(
        glin_prop=glin_prop,
        a_min=a_min_glin,
        a_max=a_max_glin,
        a_k=a_k_glin,
        neg_weight_scaled=w_negative * glin_ratio,
        glin_weight=w_glin,
        global_start_index=coll_iters,
        global_total=total_iters,
        max_iterations=max_iter_glin,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    mkm = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=best_coll,
        a_glin=best_glin,
    )
    neg_s, gb, cb = calc_metrics_mkm(mkm)
    q = calc_quality_score(neg_s, gb, cb, w_negative, w_glin, w_coll)
    return {
        "Q": float(q),
        "negative_share": float(neg_s),
        "glin_bad_share": float(gb),
        "coll_bad_share": float(cb),
        "time_sec": float(elapsed),
        "bf_evals": int(coll_iters + glin_iters),
        "coll_iters": int(coll_iters),
        "glin_iters": int(glin_iters),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BF vs GA: бенчмарк и устойчивость (skv621).")
    p.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las относительно корня проекта или абсолютный.",
    )
    p.add_argument("--config-dir", default="config")
    p.add_argument("--output-dir", default="outputs/experiments/skv621_bf_ga")
    p.add_argument("--w-negative", type=float, default=0.7)
    p.add_argument("--w-glin", type=float, default=0.3)
    p.add_argument("--w-coll", type=float, default=0.3)
    p.add_argument("--benchmark", action="store_true", help="Полный BF, GA, BF с бюджетом оценок как у GA")
    p.add_argument("--stability", action="store_true", help="Сдвиг границ: разброс Q BF vs GA")
    p.add_argument("--all", action="store_true", help="benchmark + stability")
    p.add_argument(
        "--shift-fracs",
        default="-0.04,-0.02,0,0.02,0.04",
        help="Доли сдвига границ (через запятую) для stability.",
    )
    p.add_argument("--seeds", default="42,43,44", help="Сиды GA для stability (через запятую).")
    p.add_argument("--population-size", type=int, default=220)
    p.add_argument("--ngen", type=int, default=110)
    p.add_argument("--cxpb", type=float, default=0.6)
    p.add_argument("--mutpb", type=float, default=0.25)
    p.add_argument("--indpb", type=float, default=0.1)
    p.add_argument("--tournsize", type=int, default=3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument(
        "--stability-max-iter-coll",
        type=int,
        default=0,
        help="В режиме --stability: лимит итераций COLL (0 = полный перебор по сетке).",
    )
    p.add_argument(
        "--stability-max-iter-glin",
        type=int,
        default=0,
        help="В режиме --stability: лимит итераций GLIN (0 = полный перебор).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    do_bench = args.benchmark or args.all
    do_stab = args.stability or args.all
    if not do_bench and not do_stab:
        print("Укажите --benchmark, --stability или --all")
        sys.exit(1)

    las_path = resolve_path(args.las, PROJECT_ROOT)
    config_dir = resolve_path(args.config_dir, PROJECT_ROOT)
    out_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, is_coll, is_glin, coll_prop, glin_prop, _litho_raw = load_mkm_from_las(
        las_path, verbose=True
    )
    if len(coll_prop) == 0 or len(glin_prop) == 0:
        raise ValueError("Нужны и коллекторы, и глины в LAS.")

    a_min_coll0 = np.loadtxt(config_dir / "a_min_coll.in")
    a_max_coll0 = np.loadtxt(config_dir / "a_max_coll.in")
    a_k_coll = np.loadtxt(config_dir / "a_k_coll.in")
    a_min_glin0 = np.loadtxt(config_dir / "a_min_glin.in")
    a_max_glin0 = np.loadtxt(config_dir / "a_max_glin.in")
    a_k_glin = np.loadtxt(config_dir / "a_k_glin.in")
    for arr, name in (
        (a_min_coll0, "A_min_coll"),
        (a_max_coll0, "A_max_coll"),
        (a_min_glin0, "A_min_glin"),
        (a_max_glin0, "A_max_glin"),
    ):
        validate_matrix_shape(arr, name)
    validate_k_shape(a_k_coll, "A_k_coll")
    validate_k_shape(a_k_glin, "A_k_glin")

    coll_grids = build_value_grids(a_min_coll0, a_max_coll0, a_k_coll)
    glin_grids = build_value_grids(a_min_glin0, a_max_glin0, a_k_glin)
    total_coll = prod(len(g) for g in coll_grids)
    total_glin = prod(len(g) for g in glin_grids)

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
        seed=42,
    )

    if do_bench:
        bench_rows: list[dict[str, object]] = []
        print("=== Бенчмарк (границы без сдвига) ===")
        ga_res = run_ga_once(
            data=data,
            is_coll=is_coll,
            is_glin=is_glin,
            coll_prop=coll_prop,
            glin_prop=glin_prop,
            a_min_coll=a_min_coll0,
            a_max_coll=a_max_coll0,
            a_min_glin=a_min_glin0,
            a_max_glin=a_max_glin0,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            ga_params=ga_params,
        )
        print(f"GA: Q={ga_res['Q']:.6f}, evals={ga_res['ga_fitness_evals']}, time={ga_res['time_sec']:.2f}s")
        bench_rows.append(
            {
                "method": "ga_full",
                "shift_frac": 0.0,
                "seed": ga_params.seed,
                "Q": ga_res["Q"],
                "negative_share": ga_res["negative_share"],
                "glin_bad_share": ga_res["glin_bad_share"],
                "coll_bad_share": ga_res["coll_bad_share"],
                "time_sec": ga_res["time_sec"],
                "evals": ga_res["ga_fitness_evals"],
            }
        )

        print("Полный bruteforce (тихий режим)...")
        bf_full = run_bf_once(
            data=data,
            is_coll=is_coll,
            is_glin=is_glin,
            coll_prop=coll_prop,
            glin_prop=glin_prop,
            a_min_coll=a_min_coll0,
            a_max_coll=a_max_coll0,
            a_k_coll=a_k_coll,
            a_min_glin=a_min_glin0,
            a_max_glin=a_max_glin0,
            a_k_glin=a_k_glin,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            max_iter_coll=None,
            max_iter_glin=None,
        )
        print(f"BF full: Q={bf_full['Q']:.6f}, evals={bf_full['bf_evals']}, time={bf_full['time_sec']:.2f}s")
        bench_rows.append(
            {
                "method": "bf_full",
                "shift_frac": 0.0,
                "seed": "",
                "Q": bf_full["Q"],
                "negative_share": bf_full["negative_share"],
                "glin_bad_share": bf_full["glin_bad_share"],
                "coll_bad_share": bf_full["coll_bad_share"],
                "time_sec": bf_full["time_sec"],
                "evals": bf_full["bf_evals"],
            }
        )

        ga_nevals = int(ga_res["ga_fitness_evals"])
        half = max(1, ga_nevals // 2)
        max_c = min(total_coll, half)
        max_g = min(total_glin, max(1, ga_nevals - max_c))
        print(f"BF с бюджетом оценок ~{max_c + max_g} (целевой бюджет GA {ga_nevals})...")
        bf_budget = run_bf_once(
            data=data,
            is_coll=is_coll,
            is_glin=is_glin,
            coll_prop=coll_prop,
            glin_prop=glin_prop,
            a_min_coll=a_min_coll0,
            a_max_coll=a_max_coll0,
            a_k_coll=a_k_coll,
            a_min_glin=a_min_glin0,
            a_max_glin=a_max_glin0,
            a_k_glin=a_k_glin,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            max_iter_coll=max_c,
            max_iter_glin=max_g,
        )
        print(f"BF budget: Q={bf_budget['Q']:.6f}, evals={bf_budget['bf_evals']}, time={bf_budget['time_sec']:.2f}s")
        bench_rows.append(
            {
                "method": "bf_budget_matched",
                "shift_frac": 0.0,
                "seed": "",
                "Q": bf_budget["Q"],
                "negative_share": bf_budget["negative_share"],
                "glin_bad_share": bf_budget["glin_bad_share"],
                "coll_bad_share": bf_budget["coll_bad_share"],
                "time_sec": bf_budget["time_sec"],
                "evals": bf_budget["bf_evals"],
            }
        )

        bench_path = out_dir / "benchmark_skv621.csv"
        keys = [
            "method",
            "shift_frac",
            "seed",
            "Q",
            "negative_share",
            "glin_bad_share",
            "coll_bad_share",
            "time_sec",
            "evals",
        ]
        with bench_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(bench_rows)
        print(f"Сохранено: {bench_path}")

    if do_stab:
        stab_max_c = args.stability_max_iter_coll if args.stability_max_iter_coll > 0 else None
        stab_max_g = args.stability_max_iter_glin if args.stability_max_iter_glin > 0 else None
        bf_method_label = "bf_full" if stab_max_c is None and stab_max_g is None else "bf_capped"

        fracs = [float(x.strip()) for x in args.shift_fracs.split(",") if x.strip()]
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        stab_path = out_dir / "stability_skv621.csv"
        fieldnames = [
            "shift_frac",
            "method",
            "seed",
            "Q",
            "negative_share",
            "glin_bad_share",
            "coll_bad_share",
            "time_sec",
            "evals",
        ]
        with stab_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frac in fracs:
                ac_lo, ac_hi = translate_bounds(a_min_coll0, a_max_coll0, frac)
                ag_lo, ag_hi = translate_bounds(a_min_glin0, a_max_glin0, frac)
                print(f"=== Stability shift_frac={frac} ===")
                bf_r = run_bf_once(
                    data=data,
                    is_coll=is_coll,
                    is_glin=is_glin,
                    coll_prop=coll_prop,
                    glin_prop=glin_prop,
                    a_min_coll=ac_lo,
                    a_max_coll=ac_hi,
                    a_k_coll=a_k_coll,
                    a_min_glin=ag_lo,
                    a_max_glin=ag_hi,
                    a_k_glin=a_k_glin,
                    w_negative=args.w_negative,
                    w_glin=args.w_glin,
                    w_coll=args.w_coll,
                    max_iter_coll=stab_max_c,
                    max_iter_glin=stab_max_g,
                )
                writer.writerow(
                    {
                        "shift_frac": frac,
                        "method": bf_method_label,
                        "seed": "",
                        "Q": bf_r["Q"],
                        "negative_share": bf_r["negative_share"],
                        "glin_bad_share": bf_r["glin_bad_share"],
                        "coll_bad_share": bf_r["coll_bad_share"],
                        "time_sec": bf_r["time_sec"],
                        "evals": bf_r["bf_evals"],
                    }
                )
                f.flush()
                print(f"  BF Q={bf_r['Q']:.6f}")

                for sd in seeds:
                    gp = GAParams(
                        population_size=args.population_size,
                        ngen=args.ngen,
                        cxpb=args.cxpb,
                        mutpb=args.mutpb,
                        indpb=args.indpb,
                        tournsize=args.tournsize,
                        patience=args.patience,
                        min_delta=args.min_delta,
                        n_jobs=1,
                        seed=sd,
                    )
                    gr = run_ga_once(
                        data=data,
                        is_coll=is_coll,
                        is_glin=is_glin,
                        coll_prop=coll_prop,
                        glin_prop=glin_prop,
                        a_min_coll=ac_lo,
                        a_max_coll=ac_hi,
                        a_min_glin=ag_lo,
                        a_max_glin=ag_hi,
                        w_negative=args.w_negative,
                        w_glin=args.w_glin,
                        w_coll=args.w_coll,
                        ga_params=gp,
                    )
                    writer.writerow(
                        {
                            "shift_frac": frac,
                            "method": "ga",
                            "seed": sd,
                            "Q": gr["Q"],
                            "negative_share": gr["negative_share"],
                            "glin_bad_share": gr["glin_bad_share"],
                            "coll_bad_share": gr["coll_bad_share"],
                            "time_sec": gr["time_sec"],
                            "evals": gr["ga_fitness_evals"],
                        }
                    )
                    f.flush()
                    print(f"  GA seed={sd} Q={gr['Q']:.6f}")
        print(f"Сохранено: {stab_path}")


if __name__ == "__main__":
    main()
