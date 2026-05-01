#!/usr/bin/env python3
"""Сравнение интервального bruteforce и интервального GA для дипломного отчёта."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# корень проекта = родитель каталога experiments/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from mkm_interval_optimizer import (  # noqa: E402
    IntervalOptimizationResult,
    IntervalOptimizationSummary,
    coarsen_k_matrix,
    run_interval_bruteforce,
    run_interval_ga,
    write_interval_results_csv,
)


def _summary_to_row(
    summary: IntervalOptimizationSummary,
    *,
    method_label: str,
    coarse_factor: float,
    max_iterations: int,
) -> dict[str, object]:
    return {
        "method": method_label,
        "Q": summary.quality_score,
        "negative_share": summary.negative_share,
        "glin_bad_share": summary.glin_bad_share,
        "coll_bad_share": summary.coll_bad_share,
        "time_sec": summary.total_time_sec,
        "evals": summary.total_evals,
        "intervals": len(summary.interval_results),
        "invalid_count": summary.total_invalid_count,
        "generations": summary.total_generations,
        "coarse_factor": coarse_factor,
        "max_iterations": max_iterations,
    }


def _write_interval_comparison_csv(
    csv_path: Path,
    *,
    ga_results: list[IntervalOptimizationResult],
    bf_results: list[IntervalOptimizationResult],
    bf_label: str,
) -> None:
    ga_by_id = {item.interval_id: item for item in ga_results}
    bf_by_id = {item.interval_id: item for item in bf_results}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "interval_id",
                "lithotype",
                "depth_start",
                "depth_end",
                "sample_count",
                "ga_local_score",
                "bf_local_score",
                "delta_local_score",
                "ga_negative_share",
                "bf_negative_share",
                "delta_negative_share",
                "ga_bad_share",
                "bf_bad_share",
                "delta_bad_share",
                "bf_label",
            ],
        )
        writer.writeheader()
        for interval_id in sorted(ga_by_id):
            ga_item = ga_by_id[interval_id]
            bf_item = bf_by_id[interval_id]
            writer.writerow(
                {
                    "interval_id": interval_id,
                    "lithotype": ga_item.lithotype,
                    "depth_start": ga_item.depth_start,
                    "depth_end": ga_item.depth_end,
                    "sample_count": ga_item.sample_count,
                    "ga_local_score": ga_item.local_score,
                    "bf_local_score": bf_item.local_score,
                    "delta_local_score": bf_item.local_score - ga_item.local_score,
                    "ga_negative_share": ga_item.negative_share,
                    "bf_negative_share": bf_item.negative_share,
                    "delta_negative_share": bf_item.negative_share - ga_item.negative_share,
                    "ga_bad_share": ga_item.bad_share,
                    "bf_bad_share": bf_item.bad_share,
                    "delta_bad_share": bf_item.bad_share - ga_item.bad_share,
                    "bf_label": bf_label,
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Интервальный BF vs GA: время, качество и операции.")
    p.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las относительно корня проекта или абсолютный.",
    )
    p.add_argument("--config-dir", default="config")
    p.add_argument("--output-dir", default="outputs/experiments/skv621_bf_ga")
    p.add_argument("--w-negative", type=float, default=0.8)
    p.add_argument("--w-glin", type=float, default=0.1)
    p.add_argument("--w-coll", type=float, default=0.1)
    p.add_argument(
        "--coarse-factors",
        default="2,3,4,6",
        help="Список коэффициентов огрубления BF через запятую.",
    )
    p.add_argument(
        "--coarse-max-iterations",
        type=int,
        default=0,
        help="Дополнительный лимит итераций на интервал для coarse BF (0 = без лимита).",
    )
    p.add_argument("--population-size", type=int, default=DEFAULT_GA_POPULATION_SIZE)
    p.add_argument("--ngen", type=int, default=110)
    p.add_argument("--cxpb", type=float, default=DEFAULT_GA_CXPB)
    p.add_argument("--mutpb", type=float, default=DEFAULT_GA_MUTPB)
    p.add_argument("--indpb", type=float, default=DEFAULT_GA_INDPB)
    p.add_argument("--tournsize", type=int, default=DEFAULT_GA_TOURNSIZE)
    p.add_argument("--patience", type=int, default=DEFAULT_GA_PATIENCE)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--seed", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    las_path = resolve_path(args.las, PROJECT_ROOT)
    config_dir = resolve_path(args.config_dir, PROJECT_ROOT)
    out_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, _is_coll, _is_glin, _coll_prop, _glin_prop, _litho_raw = load_mkm_from_las(
        las_path, verbose=True
    )
    intervals = split_lithotype_intervals(data)

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
    print("=== Бенчмарк интервальных методов ===")
    print(f"Интервалов литологии: {len(intervals)}")

    benchmark_rows: list[dict[str, object]] = []

    ga_summary = run_interval_ga(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll0,
        a_max_coll=a_max_coll0,
        a_min_glin=a_min_glin0,
        a_max_glin=a_max_glin0,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        ga_params=ga_params,
        verbose=False,
    )
    benchmark_rows.append(
        _summary_to_row(
            ga_summary,
            method_label="ga_interval",
            coarse_factor=1.0,
            max_iterations=0,
        )
    )
    print(
        f"GA: Q={ga_summary.quality_score:.6f}, "
        f"time={ga_summary.total_time_sec:.2f}s, evals={ga_summary.total_evals}"
    )
    write_interval_results_csv(ga_summary.interval_results, out_dir / "intervals_ga.csv")

    bf_full_summary = run_interval_bruteforce(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll0,
        a_max_coll=a_max_coll0,
        a_k_coll=a_k_coll,
        a_min_glin=a_min_glin0,
        a_max_glin=a_max_glin0,
        a_k_glin=a_k_glin,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        max_iterations=None,
        verbose=False,
    )
    benchmark_rows.append(
        _summary_to_row(
            bf_full_summary,
            method_label="bf_full_interval",
            coarse_factor=1.0,
            max_iterations=0,
        )
    )
    print(
        f"BF full: Q={bf_full_summary.quality_score:.6f}, "
        f"time={bf_full_summary.total_time_sec:.2f}s, evals={bf_full_summary.total_evals}"
    )
    write_interval_results_csv(bf_full_summary.interval_results, out_dir / "intervals_bf_full.csv")

    bf_eval_time = bf_full_summary.total_time_sec / max(1, bf_full_summary.total_evals)
    target_total_evals = int(round(ga_summary.total_time_sec / max(bf_eval_time, 1e-12)))
    matched_budget = max(1, int(round(target_total_evals / max(1, len(intervals)))))
    bf_budget_summary = run_interval_bruteforce(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll0,
        a_max_coll=a_max_coll0,
        a_k_coll=a_k_coll,
        a_min_glin=a_min_glin0,
        a_max_glin=a_max_glin0,
        a_k_glin=a_k_glin,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        max_iterations=matched_budget,
        verbose=False,
    )
    benchmark_rows.append(
        _summary_to_row(
            bf_budget_summary,
            method_label="bf_budget_time_matched",
            coarse_factor=1.0,
            max_iterations=matched_budget,
        )
    )
    print(
        f"BF budget matched: Q={bf_budget_summary.quality_score:.6f}, "
        f"time={bf_budget_summary.total_time_sec:.2f}s, evals={bf_budget_summary.total_evals}, "
        f"max_iterations={matched_budget}"
    )
    write_interval_results_csv(bf_budget_summary.interval_results, out_dir / "intervals_bf_budget_time_matched.csv")

    coarse_summaries: list[tuple[float, IntervalOptimizationSummary]] = []
    coarse_factors = [float(item.strip()) for item in args.coarse_factors.split(",") if item.strip()]
    coarse_limit = args.coarse_max_iterations if args.coarse_max_iterations > 0 else None

    for factor in coarse_factors:
        coarse_summary = run_interval_bruteforce(
            data=data,
            intervals=intervals,
            a_min_coll=a_min_coll0,
            a_max_coll=a_max_coll0,
            a_k_coll=coarsen_k_matrix(a_k_coll, factor),
            a_min_glin=a_min_glin0,
            a_max_glin=a_max_glin0,
            a_k_glin=coarsen_k_matrix(a_k_glin, factor),
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
            max_iterations=coarse_limit,
            verbose=False,
        )
        coarse_summaries.append((factor, coarse_summary))
        benchmark_rows.append(
            _summary_to_row(
                coarse_summary,
                method_label=f"bf_coarse_f{factor:g}",
                coarse_factor=factor,
                max_iterations=args.coarse_max_iterations,
            )
        )
        print(
            f"BF coarse factor={factor:g}: Q={coarse_summary.quality_score:.6f}, "
            f"time={coarse_summary.total_time_sec:.2f}s, evals={coarse_summary.total_evals}"
        )
        suffix = str(factor).replace(".", "_")
        write_interval_results_csv(coarse_summary.interval_results, out_dir / f"intervals_bf_coarse_{suffix}.csv")

    matched_factor, matched_summary = min(
        coarse_summaries,
        key=lambda item: abs(item[1].total_time_sec - ga_summary.total_time_sec),
    )
    print(
        f"Наиболее близкий по времени coarse BF: factor={matched_factor:g}, "
        f"time={matched_summary.total_time_sec:.2f}s, GA time={ga_summary.total_time_sec:.2f}s"
    )

    _write_interval_comparison_csv(
        out_dir / "interval_comparison_bf_full_vs_ga.csv",
        ga_results=ga_summary.interval_results,
        bf_results=bf_full_summary.interval_results,
        bf_label="bf_full_interval",
    )
    _write_interval_comparison_csv(
        out_dir / "interval_comparison_bf_matched_vs_ga.csv",
        ga_results=ga_summary.interval_results,
        bf_results=bf_budget_summary.interval_results,
        bf_label="bf_budget_time_matched",
    )

    bench_path = out_dir / "benchmark_skv621.csv"
    with bench_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "method",
                "Q",
                "negative_share",
                "glin_bad_share",
                "coll_bad_share",
                "time_sec",
                "evals",
                "intervals",
                "invalid_count",
                "generations",
                "coarse_factor",
                "max_iterations",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"Сохранено: {bench_path}")


if __name__ == "__main__":
    main()
