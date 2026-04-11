"""CLI: перебор матриц МКМ для произвольного LAS."""

from __future__ import annotations

import argparse
import time
from math import prod
from pathlib import Path

import numpy as np

from mkm_bruteforce_engine import (
    brute_force_best_coll,
    brute_force_best_glin,
    build_value_grids,
)
from mkm_core import (
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    calc_mkm_model,
    calc_metrics_mkm,
    default_mkm_artifact_paths,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    validate_k_shape,
    validate_matrix_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Перебор матриц A_coll/A_glin по сетке (a_k_*.in), построение лучшей МКМ, "
            "метрики и график по глубине."
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
        help="Четыре кривые-свойства (иначе автоподбор после POTA THOR RHOB).",
    )
    parser.add_argument("--config-dir", default="config", help="Каталог с a_*.in.")

    parser.add_argument("--a-min-coll", default="a_min_coll.in")
    parser.add_argument("--a-max-coll", default="a_max_coll.in")
    parser.add_argument("--a-k-coll", default="a_k_coll.in")
    parser.add_argument("--a-min-glin", default="a_min_glin.in")
    parser.add_argument("--a-max-glin", default="a_max_glin.in")
    parser.add_argument("--a-k-glin", default="a_k_glin.in")

    parser.add_argument("--best-coll-out", default="", help="Пусто → outputs/matrices/<stem>_coll_bf.out")
    parser.add_argument("--best-glin-out", default="", help="Пусто → outputs/matrices/<stem>_glin_bf.out")
    parser.add_argument("--plot-png", default="", help="Пусто → outputs/plots/<stem>_mkm_bf.png")
    parser.add_argument("--save-mkm", default="")

    parser.add_argument(
        "--w-negative",
        type=float,
        default=0.7,
        help="Вес доли отрицательных (как у mkm_run_ga.py для сопоставимости с GA).",
    )
    parser.add_argument("--w-glin", type=float, default=0.3, help="Вес метрики глин (как у GA).")
    parser.add_argument("--w-coll", type=float, default=0.3, help="Вес метрики коллекторов (как у GA).")
    parser.add_argument("--max-iter-coll", type=int, default=0, help="0 = без ограничения.")
    parser.add_argument("--max-iter-glin", type=int, default=0, help="0 = без ограничения.")
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
    if len(coll_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 1 (коллекторы).")
    if len(glin_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 2 (глины).")

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

    coll_grids = build_value_grids(a_min_coll, a_max_coll, a_k_coll)
    glin_grids = build_value_grids(a_min_glin, a_max_glin, a_k_glin)
    total_coll = prod(len(g) for g in coll_grids)
    total_glin = prod(len(g) for g in glin_grids)
    max_iter_coll = args.max_iter_coll if args.max_iter_coll > 0 else None
    max_iter_glin = args.max_iter_glin if args.max_iter_glin > 0 else None
    total_coll_effective = min(total_coll, max_iter_coll) if max_iter_coll is not None else total_coll
    total_glin_effective = min(total_glin, max_iter_glin) if max_iter_glin is not None else total_glin
    total_iterations = total_coll_effective + total_glin_effective

    coll_ratio = len(coll_prop) / (len(coll_prop) + len(glin_prop))
    glin_ratio = len(glin_prop) / (len(coll_prop) + len(glin_prop))
    coll_neg_weight_scaled = args.w_negative * coll_ratio
    glin_neg_weight_scaled = args.w_negative * glin_ratio

    print("Старт брутфорса матриц.")
    print(
        f"Всего итераций: {total_iterations} (COLL={total_coll_effective}, "
        f"GLIN={total_glin_effective})"
    )
    print(
        "Критерий минимизации: "
        "w_negative * negative_share + w_glin * glin_bad + w_coll * coll_bad"
    )
    print(
        f"Весовые коэффициенты: w_negative={args.w_negative}, "
        f"w_glin={args.w_glin}, w_coll={args.w_coll}"
    )

    brute_start = time.perf_counter()

    verbose_bf = not args.quiet
    best_coll, best_coll_score, _, _, coll_iters, coll_invalid = brute_force_best_coll(
        coll_prop=coll_prop,
        a_min=a_min_coll,
        a_max=a_max_coll,
        a_k=a_k_coll,
        neg_weight_scaled=coll_neg_weight_scaled,
        coll_weight=args.w_coll,
        global_start_index=0,
        global_total=total_iterations,
        max_iterations=max_iter_coll,
        verbose=verbose_bf,
    )

    best_glin, best_glin_score, _, _, glin_iters, glin_invalid = brute_force_best_glin(
        glin_prop=glin_prop,
        a_min=a_min_glin,
        a_max=a_max_glin,
        a_k=a_k_glin,
        neg_weight_scaled=glin_neg_weight_scaled,
        glin_weight=args.w_glin,
        global_start_index=coll_iters,
        global_total=total_iterations,
        max_iterations=max_iter_glin,
        verbose=verbose_bf,
    )

    brute_elapsed_sec = time.perf_counter() - brute_start

    best_mkm_model = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=best_coll,
        a_glin=best_glin,
    )

    negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(best_mkm_model)
    total_score = (
        args.w_negative * negative_share
        + args.w_glin * glin_bad_share
        + args.w_coll * coll_bad_share
    )

    stem = las_path.stem
    defaults = default_mkm_artifact_paths(project_root, stem, "bf")
    best_coll_path = (
        resolve_path(args.best_coll_out, project_root)
        if args.best_coll_out
        else defaults["coll"]
    )
    best_glin_path = (
        resolve_path(args.best_glin_out, project_root)
        if args.best_glin_out
        else defaults["glin"]
    )
    plot_path = (
        resolve_path(args.plot_png, project_root)
        if args.plot_png
        else defaults["plot"]
    )

    best_coll_path.parent.mkdir(parents=True, exist_ok=True)
    best_glin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(best_coll_path, best_coll, fmt="%.12g")
    np.savetxt(best_glin_path, best_glin, fmt="%.12g")

    save_mkm_plot(
        best_mkm_model,
        plot_path,
        litho_raw=litho_raw,
        litho_mnem=args.litho,
    )

    if args.save_mkm:
        save_mkm_path = resolve_path(args.save_mkm, project_root)
        np.save(save_mkm_path, best_mkm_model)
        print(f"Лучшая МКМ-модель сохранена в: {save_mkm_path}")

    print("\nБрутфорс завершен.")
    print(f"Время перебора (только brute force): {brute_elapsed_sec:.3f} сек")
    print(
        f"Итерации COLL: {coll_iters}, сингулярных матриц: {coll_invalid}; "
        f"итерации GLIN: {glin_iters}, сингулярных матриц: {glin_invalid}"
    )
    print(f"Лучший локальный score COLL: {best_coll_score:.8f}")
    print(f"Лучший локальный score GLIN: {best_glin_score:.8f}")
    print(f"Итоговый score лучшей пары: {total_score:.8f}")
    print("\nЛучшая матрица COLL:")
    print(best_coll)
    print("\nЛучшая матрица GLIN:")
    print(best_glin)

    print("\nМетрики лучшей МКМ-модели:")
    print(f"1) Доля отрицательных значений: {negative_share:.8f}")
    print(f"2) Доля глин, где сумма глин < 30%: {glin_bad_share:.8f}")
    print(f"3) Доля коллекторов, где сумма глин > 30%: {coll_bad_share:.8f}")

    print(f"\nЛучшая матрица COLL сохранена в: {best_coll_path}")
    print(f"Лучшая матрица GLIN сохранена в: {best_glin_path}")
    print(f"График лучшей МКМ-модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
