"""CLI: интервальный bruteforce матриц МКМ для произвольного LAS."""

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
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_interval_optimizer import (
    apply_k_splitting,
    run_interval_bruteforce,
    save_interval_matrices_npz,
    write_interval_results_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Интервальный перебор матриц A_coll/A_glin по непрерывным интервалам литологии "
            "с построением общей МКМ, метрик и графика по глубине."
        )
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Корень проекта (по умолчанию каталог mkm_core.py).",
    )
    parser.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las относительно корня проекта или абсолютный (по умолчанию основная скважина проекта).",
    )
    parser.add_argument("--depth", default="DEPT", help="Мнемоника глубины.")
    parser.add_argument("--litho", default="LITO", help="Мнемоника литологии.")
    parser.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства (иначе по умолчанию: POTA THOR RHOB WNKT).",
    )
    parser.add_argument("--config-dir", default="config", help="Каталог с a_*.in.")

    parser.add_argument("--a-min-coll", default="a_min_coll.in")
    parser.add_argument("--a-max-coll", default="a_max_coll.in")
    parser.add_argument("--a-k-coll", default="a_k_coll.in")
    parser.add_argument("--a-min-glin", default="a_min_glin.in")
    parser.add_argument("--a-max-glin", default="a_max_glin.in")
    parser.add_argument("--a-k-glin", default="a_k_glin.in")

    parser.add_argument("--plot-png", default="", help="Пусто → outputs/plots/<stem>_mkm_bf.png")
    parser.add_argument("--save-mkm", default="")
    parser.add_argument(
        "--interval-matrices-out",
        default="",
        help="Пусто → outputs/matrices/<stem>_intervals_bf.npz",
    )
    parser.add_argument(
        "--interval-summary-csv",
        default="",
        help="Пусто → outputs/experiments/<stem>_bf_interval_metrics.csv",
    )

    parser.add_argument(
        "--w-negative",
        type=float,
        default=0.8,
        help="Вес доли отрицательных.",
    )
    parser.add_argument("--w-glin", type=float, default=0.1, help="Вес метрики глин.")
    parser.add_argument("--w-coll", type=float, default=0.1, help="Вес метрики коллекторов.")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 = без ограничения для каждого интервала.")
    parser.add_argument(
        "--splitting",
        type=int,
        choices=(2, 3, 4, 5),
        default=5,
        help=(
            "Число узлов linspace для каждого параметра матрицы, где в a_k_coll / a_k_glin не 1: "
            "все такие k заменяются на это значение (2..5). Единицы в a_k не меняются."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать каждую итерацию перебора (для больших сеток).",
    )
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
    data, is_coll, is_glin, coll_prop, glin_prop, litho_raw = load_mkm_from_las(
        las_path,
        depth_mnem=args.depth,
        litho_mnem=args.litho,
        prop_mnems=props,
        verbose=True,
    )
    a_min_coll = np.loadtxt(config_dir / args.a_min_coll)
    a_max_coll = np.loadtxt(config_dir / args.a_max_coll)
    a_k_coll = np.loadtxt(config_dir / args.a_k_coll)
    a_min_glin = np.loadtxt(config_dir / args.a_min_glin)
    a_max_glin = np.loadtxt(config_dir / args.a_max_glin)
    a_k_glin = np.loadtxt(config_dir / args.a_k_glin)

    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")
    validate_k_shape(a_k_coll, "A_k_coll")
    validate_k_shape(a_k_glin, "A_k_glin")

    intervals = split_lithotype_intervals(data)
    a_k_coll_eff = apply_k_splitting(a_k_coll, args.splitting)
    a_k_glin_eff = apply_k_splitting(a_k_glin, args.splitting)
    max_iterations = args.max_iterations if args.max_iterations > 0 else None

    print("Старт интервального брутфорса матриц.")
    print(
        f"Интервалов литологии: {len(intervals)} "
        f"(коллектор={sum(i.lithotype == 1 for i in intervals)}, "
        f"глина={sum(i.lithotype == 2 for i in intervals)})"
    )
    print(
        f"Весовые коэффициенты: w_negative={args.w_negative}, "
        f"w_glin={args.w_glin}, w_coll={args.w_coll}"
    )
    print(f"Splitting (a_k, ячейки != 1): {args.splitting}")

    summary = run_interval_bruteforce(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll,
        a_max_coll=a_max_coll,
        a_k_coll=a_k_coll_eff,
        a_min_glin=a_min_glin,
        a_max_glin=a_max_glin,
        a_k_glin=a_k_glin_eff,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        max_iterations=max_iterations,
        verbose=not args.quiet,
    )

    stem = las_path.stem
    defaults = default_mkm_artifact_paths(project_root, stem, "bf")
    plot_path = (
        resolve_path(args.plot_png, project_root)
        if args.plot_png
        else defaults["plot"]
    )
    interval_matrices_path = (
        resolve_path(args.interval_matrices_out, project_root)
        if args.interval_matrices_out
        else project_root / "outputs" / "matrices" / f"{stem}_intervals_bf.npz"
    )
    interval_csv_path = (
        resolve_path(args.interval_summary_csv, project_root)
        if args.interval_summary_csv
        else project_root / "outputs" / "experiments" / f"{stem}_bf_interval_metrics.csv"
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

    print("\nИнтервальный брутфорс завершен.")
    print(f"Суммарное время перебора: {summary.total_time_sec:.3f} сек")
    print(f"Всего интервалов: {len(intervals)}")
    print(f"Всего оценок: {summary.total_evals}")
    print(f"Сингулярных матриц: {summary.total_invalid_count}")
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
