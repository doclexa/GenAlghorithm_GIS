"""CLI: интервальный генетический поиск матриц МКМ для произвольного LAS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mkm_core import (
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    default_mkm_artifact_paths,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    scale_mkm_model_for_metrics,
    split_lithotype_intervals,
    validate_matrix_shape,
)
from mkm_ga_engine import (
    DEFAULT_GA_CXPB,
    DEFAULT_GA_INDPB,
    DEFAULT_GA_MUTPB,
    DEFAULT_GA_PATIENCE,
    DEFAULT_GA_POPULATION_SIZE,
    DEFAULT_GA_TOURNSIZE,
    GAParams,
)
from mkm_interval_optimizer import (
    run_interval_ga,
    save_interval_matrices_npz,
    write_interval_results_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Интервальный генетический поиск матриц A_coll/A_glin для МКМ по указанному .las: "
            "метрики качества, сохранение интервальных матриц и график компонент по глубине."
        )
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Корень проекта (по умолчанию каталог, где лежит mkm_core.py).",
    )
    parser.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las относительно корня проекта или абсолютный (по умолчанию основная скважина проекта).",
    )
    parser.add_argument(
        "--depth",
        default="DEPT",
        help="Мнемоника кривой глубины (по умолчанию DEPT).",
    )
    parser.add_argument(
        "--litho",
        default="LITO",
        help="Мнемоника литологии: 1=коллектор, остальное→2=глина (по умолчанию LITO).",
    )
    parser.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства для МКМ (иначе автоматически: POTA THOR RHOB + четвёртая из типичного списка).",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Каталог с a_min_*.in, a_max_*.in относительно корня проекта.",
    )
    parser.add_argument("--a-min-coll", default="a_min_coll.in")
    parser.add_argument("--a-max-coll", default="a_max_coll.in")
    parser.add_argument("--a-min-glin", default="a_min_glin.in")
    parser.add_argument("--a-max-glin", default="a_max_glin.in")

    parser.add_argument(
        "--plot-png",
        default="",
        help="График МКМ (пусто = outputs/plots/<stem>_mkm_ga.png).",
    )
    parser.add_argument("--save-mkm", default="", help="Опционально: путь для сохранения МКМ в .npy.")
    parser.add_argument(
        "--interval-matrices-out",
        default="",
        help="Пусто → outputs/matrices/<stem>_intervals_ga.npz",
    )
    parser.add_argument(
        "--interval-summary-csv",
        default="",
        help="Пусто → outputs/experiments/<stem>_ga_interval_metrics.csv",
    )

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
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else PROJECT_ROOT
    )
    las_path = resolve_path(args.las, project_root)
    config_dir = resolve_path(args.config_dir, project_root)

    props = tuple(args.props) if args.props is not None else None
    data, _is_coll, _is_glin, _coll_prop, _glin_prop, litho_raw = load_mkm_from_las(
        las_path,
        depth_mnem=args.depth,
        litho_mnem=args.litho,
        prop_mnems=props,
        verbose=True,
    )

    a_min_coll = np.loadtxt(config_dir / args.a_min_coll)
    a_max_coll = np.loadtxt(config_dir / args.a_max_coll)
    a_min_glin = np.loadtxt(config_dir / args.a_min_glin)
    a_max_glin = np.loadtxt(config_dir / args.a_max_glin)

    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    ga_params = GAParams(
        population_size=args.population_size,
        ngen=args.ngen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        indpb=args.indpb,
        tournsize=args.tournsize,
        patience=args.patience,
        min_delta=args.min_delta,
        n_jobs=max(1, args.n_jobs),
        seed=args.seed,
    )
    intervals = split_lithotype_intervals(data)

    print("Старт интервальной GA-оптимизации матриц.")
    print(
        f"Параметры GA: pop={ga_params.population_size}, ngen={ga_params.ngen}, "
        f"cxpb={ga_params.cxpb}, mutpb={ga_params.mutpb}, indpb={ga_params.indpb}, "
        f"tournsize={ga_params.tournsize}, patience={ga_params.patience}, n_jobs={ga_params.n_jobs}"
    )
    print(
        f"Целевая функция качества: Q={args.w_negative}*negative + "
        f"{args.w_glin}*glin_bad + {args.w_coll}*coll_bad"
    )
    print(
        f"Интервалов литологии: {len(intervals)} "
        f"(коллектор={sum(i.lithotype == 1 for i in intervals)}, "
        f"глина={sum(i.lithotype == 2 for i in intervals)})"
    )

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
        ga_params=ga_params,
        verbose=True,
    )

    stem = las_path.stem
    defaults = default_mkm_artifact_paths(project_root, stem, "ga")
    plot_path = (
        resolve_path(args.plot_png, project_root)
        if args.plot_png
        else defaults["plot"]
    )
    interval_matrices_path = (
        resolve_path(args.interval_matrices_out, project_root)
        if args.interval_matrices_out
        else project_root / "outputs" / "matrices" / f"{stem}_intervals_ga.npz"
    )
    interval_csv_path = (
        resolve_path(args.interval_summary_csv, project_root)
        if args.interval_summary_csv
        else project_root / "outputs" / "experiments" / f"{stem}_ga_interval_metrics.csv"
    )

    save_interval_matrices_npz(summary, interval_matrices_path)
    write_interval_results_csv(summary.interval_results, interval_csv_path)

    mkm_plot = scale_mkm_model_for_metrics(summary.mkm_model)
    save_mkm_plot(
        mkm_plot,
        plot_path,
        litho_raw=litho_raw,
        litho_mnem=args.litho,
        intervals=intervals,
    )

    if args.save_mkm:
        save_mkm_path = resolve_path(args.save_mkm, project_root)
        np.save(save_mkm_path, mkm_plot)
        print(f"Лучшая МКМ-модель сохранена в: {save_mkm_path}")

    print("\nИнтервальная GA-оптимизация завершена.")
    print(f"Суммарное время поиска: {summary.total_time_sec:.3f} сек")
    print(f"Всего интервалов: {len(intervals)}")
    print(f"Всего оценок fitness: {summary.total_evals}")
    print(f"Суммарно поколений: {summary.total_generations}")
    print(f"Итоговый Q score: {summary.quality_score:.8f}")

    print("\nМетрики лучшей МКМ-модели:")
    print(f"1) Доля отрицательных значений: {summary.negative_share:.8f}")
    print(f"2) Доля глин, где сумма глин < 30%: {summary.glin_bad_share:.8f}")
    print(f"3) Доля коллекторов, где сумма глин > 30%: {summary.coll_bad_share:.8f}")

    print(f"\nИнтервальные матрицы сохранены в: {interval_matrices_path}")
    print(f"Сводка по интервалам сохранена в: {interval_csv_path}")
    print(f"График лучшей МКМ-модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
