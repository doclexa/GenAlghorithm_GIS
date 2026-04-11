from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass

import numpy as np

from mkm_core import (
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    calc_mkm_model,
    calc_metrics_mkm,
    calc_quality_score,
    load_mkm_from_las as load_data_from_las,
    resolve_path,
    save_mkm_plot,
    validate_matrix_shape,
)
from mkm_ga_engine import GAParams, optimize_mkm_with_ga


@dataclass
class TrialResult:
    trial_id: int
    ga_params: GAParams
    search_time_sec: float
    negative_share: float
    glin_bad_share: float
    coll_bad_share: float
    quality_score: float
    feasible: bool
    better_than_baseline: bool
    coll_matrix: np.ndarray
    glin_matrix: np.ndarray


def make_trial_key(
    result: TrialResult,
    target_time: float,
    baseline_q: float,
) -> tuple[float, float, float, float, float]:
    if result.feasible:
        return (0.0, result.quality_score, result.search_time_sec, 0.0, 0.0)
    quality_excess = max(0.0, result.quality_score - baseline_q)
    time_excess = max(0.0, result.search_time_sec - target_time)
    return (1.0, quality_excess, time_excess, result.quality_score, result.search_time_sec)


def sample_ga_params(
    rng: random.Random,
    n_jobs: int,
) -> GAParams:
    return GAParams(
        population_size=rng.choice([140, 180, 220, 280, 340, 420]),
        ngen=rng.choice([70, 90, 110, 140, 180, 220]),
        cxpb=rng.choice([0.5, 0.6, 0.7]),
        mutpb=rng.choice([0.2, 0.25, 0.3, 0.35]),
        indpb=rng.choice([0.06, 0.08, 0.1, 0.12, 0.15]),
        tournsize=rng.choice([2, 3, 4]),
        patience=rng.choice([18, 25, 35, 45, 60]),
        min_delta=rng.choice([1e-6, 1e-7]),
        n_jobs=max(1, n_jobs),
        seed=rng.randint(1, 1_000_000),
    )


def format_params_short(params: GAParams) -> str:
    return (
        f"pop={params.population_size}, ngen={params.ngen}, cxpb={params.cxpb}, "
        f"mutpb={params.mutpb}, indpb={params.indpb}, tourn={params.tournsize}, "
        f"patience={params.patience}, min_delta={params.min_delta}, n_jobs={params.n_jobs}, seed={params.seed}"
    )


def trial_result_to_row(result: TrialResult) -> dict[str, object]:
    return {
        "trial_id": result.trial_id,
        "search_time_sec": f"{result.search_time_sec:.6f}",
        "quality_score": f"{result.quality_score:.10f}",
        "negative_share": f"{result.negative_share:.10f}",
        "glin_bad_share": f"{result.glin_bad_share:.10f}",
        "coll_bad_share": f"{result.coll_bad_share:.10f}",
        "feasible": int(result.feasible),
        "better_than_baseline": int(result.better_than_baseline),
        "population_size": result.ga_params.population_size,
        "ngen": result.ga_params.ngen,
        "cxpb": result.ga_params.cxpb,
        "mutpb": result.ga_params.mutpb,
        "indpb": result.ga_params.indpb,
        "tournsize": result.ga_params.tournsize,
        "patience": result.ga_params.patience,
        "min_delta": result.ga_params.min_delta,
        "n_jobs": result.ga_params.n_jobs,
        "seed": result.ga_params.seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Перебор гиперпараметров GA (mkm_run_ga) для поиска конфигурации, "
            "которая быстрее target_time и качественнее baseline."
        )
    )
    parser.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las (по умолчанию основная скважина проекта).",
    )
    parser.add_argument("--a-min-coll", default="config/a_min_coll.in", help="Путь к a_min_coll.in.")
    parser.add_argument("--a-max-coll", default="config/a_max_coll.in", help="Путь к a_max_coll.in.")
    parser.add_argument("--a-min-glin", default="config/a_min_glin.in", help="Путь к a_min_glin.in.")
    parser.add_argument("--a-max-glin", default="config/a_max_glin.in", help="Путь к a_max_glin.in.")

    parser.add_argument("--trials", type=int, default=16, help="Число trial-ов гиперпараметров.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed тюнера.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Число процессов для fitness внутри GA.")
    parser.add_argument("--target-time", type=float, default=30.0, help="Целевой лимит времени поиска (сек).")
    parser.add_argument(
        "--max-tuning-time",
        type=float,
        default=0.0,
        help="Ограничение на общее время тюнинга в секундах (0 = без ограничения).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Сколько лучших trial-ов показать в конце.")

    parser.add_argument("--w-negative", type=float, default=0.7, help="Вес доли отрицательных значений.")
    parser.add_argument("--w-glin", type=float, default=0.3, help="Вес доли плохих глин.")
    parser.add_argument("--w-coll", type=float, default=0.3, help="Вес доли плохих коллекторов.")

    parser.add_argument("--baseline-negative", type=float, default=0.04286606, help="Baseline m1.")
    parser.add_argument("--baseline-glin", type=float, default=0.06012427, help="Baseline m2.")
    parser.add_argument("--baseline-coll", type=float, default=0.00512013, help="Baseline m3.")

    parser.add_argument(
        "--report-csv",
        default="outputs/mkm_gen_tuning_report.csv",
        help="Куда сохранить таблицу trial-ов.",
    )
    parser.add_argument(
        "--best-coll-out",
        default="outputs/matrices/best_matrix_coll_gen_tuned.out",
        help="Куда сохранить лучшую матрицу коллектора.",
    )
    parser.add_argument(
        "--best-glin-out",
        default="outputs/matrices/best_matrix_glin_gen_tuned.out",
        help="Куда сохранить лучшую матрицу глины.",
    )
    parser.add_argument(
        "--best-plot-png",
        default="outputs/plots/mkm_gen_tuned_best_plot.png",
        help="Куда сохранить график лучшей МКМ-модели.",
    )
    parser.add_argument(
        "--save-best-mkm",
        default="",
        help="Необязательно: путь для сохранения лучшей МКМ-модели в .npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    baseline_q = calc_quality_score(
        negative_share=args.baseline_negative,
        glin_bad_share=args.baseline_glin,
        coll_bad_share=args.baseline_coll,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
    )

    las_path = resolve_path(args.las, PROJECT_ROOT)
    a_min_coll_path = resolve_path(args.a_min_coll, PROJECT_ROOT)
    a_max_coll_path = resolve_path(args.a_max_coll, PROJECT_ROOT)
    a_min_glin_path = resolve_path(args.a_min_glin, PROJECT_ROOT)
    a_max_glin_path = resolve_path(args.a_max_glin, PROJECT_ROOT)

    data, is_coll, is_glin, coll_prop, glin_prop, litho_raw = load_data_from_las(las_path)
    if len(coll_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 1 (коллекторы).")
    if len(glin_prop) == 0:
        raise ValueError("В данных нет строк с LITO != 1 (глины после преобразования в 2).")

    a_min_coll = np.loadtxt(a_min_coll_path)
    a_max_coll = np.loadtxt(a_max_coll_path)
    a_min_glin = np.loadtxt(a_min_glin_path)
    a_max_glin = np.loadtxt(a_max_glin_path)
    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    print("Старт тюнинга гиперпараметров GA.")
    print(
        f"Baseline: m1={args.baseline_negative:.8f}, m2={args.baseline_glin:.8f}, "
        f"m3={args.baseline_coll:.8f}, Q_baseline={baseline_q:.8f}"
    )
    print(
        f"Цель: search_time < {args.target_time:.2f}s и Q < {baseline_q:.8f}."
    )

    tried_configs: set[tuple] = set()
    trial_results: list[TrialResult] = []
    tuning_start = time.perf_counter()

    baseline_config = GAParams(
        population_size=220,
        ngen=110,
        cxpb=0.6,
        mutpb=0.25,
        indpb=0.1,
        tournsize=3,
        patience=25,
        min_delta=1e-7,
        n_jobs=max(1, args.n_jobs),
        seed=args.seed,
    )

    for trial_id in range(1, args.trials + 1):
        if trial_id == 1:
            ga_params = baseline_config
        else:
            while True:
                ga_params = sample_ga_params(rng, args.n_jobs)
                config_key = (
                    ga_params.population_size,
                    ga_params.ngen,
                    ga_params.cxpb,
                    ga_params.mutpb,
                    ga_params.indpb,
                    ga_params.tournsize,
                    ga_params.patience,
                    ga_params.min_delta,
                    ga_params.seed,
                    ga_params.n_jobs,
                )
                if config_key not in tried_configs:
                    break
        tried_configs.add(
            (
                ga_params.population_size,
                ga_params.ngen,
                ga_params.cxpb,
                ga_params.mutpb,
                ga_params.indpb,
                ga_params.tournsize,
                ga_params.patience,
                ga_params.min_delta,
                ga_params.seed,
                ga_params.n_jobs,
            )
        )

        coll_result, glin_result, search_time_sec = optimize_mkm_with_ga(
            coll_prop=coll_prop,
            glin_prop=glin_prop,
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

        mkm_model = calc_mkm_model(
            data=data,
            is_coll=is_coll,
            is_glin=is_glin,
            coll_prop=coll_prop,
            glin_prop=glin_prop,
            a_coll=coll_result.best_matrix,
            a_glin=glin_result.best_matrix,
        )
        m1, m2, m3 = calc_metrics_mkm(mkm_model)
        q_value = calc_quality_score(
            negative_share=m1,
            glin_bad_share=m2,
            coll_bad_share=m3,
            w_negative=args.w_negative,
            w_glin=args.w_glin,
            w_coll=args.w_coll,
        )
        feasible = search_time_sec < args.target_time and q_value < baseline_q
        better_than_baseline = q_value < baseline_q

        result = TrialResult(
            trial_id=trial_id,
            ga_params=ga_params,
            search_time_sec=search_time_sec,
            negative_share=m1,
            glin_bad_share=m2,
            coll_bad_share=m3,
            quality_score=q_value,
            feasible=feasible,
            better_than_baseline=better_than_baseline,
            coll_matrix=coll_result.best_matrix.copy(),
            glin_matrix=glin_result.best_matrix.copy(),
        )
        trial_results.append(result)

        verdict = "OK" if feasible else "NO"
        print(
            f"[Trial {trial_id:02d}/{args.trials}] time={search_time_sec:7.3f}s "
            f"Q={q_value:.8f} m1={m1:.8f} m2={m2:.8f} m3={m3:.8f} feasible={verdict} "
            f"| {format_params_short(ga_params)}"
        )

        if args.max_tuning_time > 0:
            tuning_elapsed = time.perf_counter() - tuning_start
            if tuning_elapsed >= args.max_tuning_time:
                print(
                    f"Остановка по лимиту max_tuning_time={args.max_tuning_time:.2f}s "
                    f"после {trial_id} trial(ов)."
                )
                break

    if not trial_results:
        raise RuntimeError("Тюнинг не запустился: нет результатов trial-ов.")

    sorted_results = sorted(
        trial_results,
        key=lambda r: make_trial_key(r, target_time=args.target_time, baseline_q=baseline_q),
    )
    best_result = sorted_results[0]

    report_path = resolve_path(args.report_csv, PROJECT_ROOT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(trial_result_to_row(best_result).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in sorted_results:
            writer.writerow(trial_result_to_row(trial))

    best_mkm = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=best_result.coll_matrix,
        a_glin=best_result.glin_matrix,
    )

    best_coll_path = resolve_path(args.best_coll_out, PROJECT_ROOT)
    best_glin_path = resolve_path(args.best_glin_out, PROJECT_ROOT)
    best_plot_path = resolve_path(args.best_plot_png, PROJECT_ROOT)
    best_coll_path.parent.mkdir(parents=True, exist_ok=True)
    best_glin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(best_coll_path, best_result.coll_matrix, fmt="%.12g")
    np.savetxt(best_glin_path, best_result.glin_matrix, fmt="%.12g")
    save_mkm_plot(best_mkm, best_plot_path, litho_raw=litho_raw, litho_mnem="LITO")

    if args.save_best_mkm:
        mkm_path = resolve_path(args.save_best_mkm, PROJECT_ROOT)
        np.save(mkm_path, best_mkm)
        print(f"Лучшая МКМ-модель сохранена в: {mkm_path}")

    print("\nТоп лучших trial-ов:")
    show_top = min(args.top_k, len(sorted_results))
    for idx in range(show_top):
        trial = sorted_results[idx]
        feasible_tag = "OK" if trial.feasible else "NO"
        print(
            f"{idx + 1:02d}) Trial={trial.trial_id:02d} Q={trial.quality_score:.8f} "
            f"time={trial.search_time_sec:.3f}s feasible={feasible_tag} "
            f"| {format_params_short(trial.ga_params)}"
        )

    total_tuning_time = time.perf_counter() - tuning_start
    delta_q = best_result.quality_score - baseline_q
    delta_m1 = best_result.negative_share - args.baseline_negative
    delta_m2 = best_result.glin_bad_share - args.baseline_glin
    delta_m3 = best_result.coll_bad_share - args.baseline_coll

    print("\nИтог тюнинга:")
    print(f"Всего trial-ов: {len(trial_results)}")
    print(f"Общее время тюнинга: {total_tuning_time:.3f} сек")
    print(f"Baseline Q: {baseline_q:.8f}")
    print(f"Лучший Q: {best_result.quality_score:.8f} (delta={delta_q:+.8f})")
    print(f"m1 (delta): {best_result.negative_share:.8f} ({delta_m1:+.8f})")
    print(f"m2 (delta): {best_result.glin_bad_share:.8f} ({delta_m2:+.8f})")
    print(f"m3 (delta): {best_result.coll_bad_share:.8f} ({delta_m3:+.8f})")
    print(f"Время поиска лучшего trial: {best_result.search_time_sec:.3f} сек")
    print(f"Цель (<{args.target_time:.2f}s и Q < baseline): {'ДОСТИГНУТА' if best_result.feasible else 'НЕ ДОСТИГНУТА'}")

    print(f"\nCSV-отчет сохранен: {report_path}")
    print(f"Лучшая матрица COLL сохранена: {best_coll_path}")
    print(f"Лучшая матрица GLIN сохранена: {best_glin_path}")
    print(f"График лучшей МКМ-модели сохранен: {best_plot_path}")


if __name__ == "__main__":
    main()
