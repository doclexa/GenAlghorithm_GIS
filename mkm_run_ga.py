"""CLI: генетический поиск матриц МКМ для произвольного LAS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mkm_core import (
    PROJECT_ROOT,
    calc_mkm_model,
    calc_metrics_mkm,
    calc_quality_score,
    default_mkm_artifact_paths,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    validate_matrix_shape,
)
from mkm_ga_engine import GAParams, optimize_mkm_with_ga


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Генетический поиск матриц A_coll/A_glin для МКМ по указанному .las: "
            "метрики качества, сохранение матриц и график компонент по глубине."
        )
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Корень проекта (по умолчанию каталог, где лежит mkm_core.py).",
    )
    parser.add_argument(
        "--las",
        default="data/las/inp.las",
        help="Путь к .las относительно корня проекта или абсолютный.",
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
        "--best-coll-out",
        default="",
        help="Куда сохранить лучшую матрицу коллектора (пусто = outputs/matrices/<stem>_coll_ga.out).",
    )
    parser.add_argument(
        "--best-glin-out",
        default="",
        help="Куда сохранить лучшую матрицу глины (пусто = outputs/matrices/<stem>_glin_ga.out).",
    )
    parser.add_argument(
        "--plot-png",
        default="",
        help="График МКМ (пусто = outputs/plots/<stem>_mkm_ga.png).",
    )
    parser.add_argument("--save-mkm", default="", help="Опционально: путь для сохранения МКМ в .npy.")

    parser.add_argument("--w-negative", type=float, default=0.7)
    parser.add_argument("--w-glin", type=float, default=0.3)
    parser.add_argument("--w-coll", type=float, default=0.3)

    parser.add_argument("--population-size", type=int, default=220)
    parser.add_argument("--ngen", type=int, default=110)
    parser.add_argument("--cxpb", type=float, default=0.6)
    parser.add_argument("--mutpb", type=float, default=0.25)
    parser.add_argument("--indpb", type=float, default=0.1)
    parser.add_argument("--tournsize", type=int, default=3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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
    data, is_coll, is_glin, coll_prop, glin_prop = load_mkm_from_las(
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

    print("Старт GA-оптимизации матриц.")
    print(
        f"Параметры GA: pop={ga_params.population_size}, ngen={ga_params.ngen}, "
        f"cxpb={ga_params.cxpb}, mutpb={ga_params.mutpb}, indpb={ga_params.indpb}, "
        f"tournsize={ga_params.tournsize}, patience={ga_params.patience}, n_jobs={ga_params.n_jobs}"
    )
    print(
        f"Целевая функция качества: Q={args.w_negative}*negative + "
        f"{args.w_glin}*glin_bad + {args.w_coll}*coll_bad"
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
        verbose=True,
    )

    best_mkm_model = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=coll_result.best_matrix,
        a_glin=glin_result.best_matrix,
    )
    negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(best_mkm_model)
    quality_score = calc_quality_score(
        negative_share=negative_share,
        glin_bad_share=glin_bad_share,
        coll_bad_share=coll_bad_share,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
    )

    stem = las_path.stem
    defaults = default_mkm_artifact_paths(project_root, stem, "ga")
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
    np.savetxt(best_coll_path, coll_result.best_matrix, fmt="%.12g")
    np.savetxt(best_glin_path, glin_result.best_matrix, fmt="%.12g")

    save_mkm_plot(best_mkm_model, plot_path)

    if args.save_mkm:
        save_mkm_path = resolve_path(args.save_mkm, project_root)
        np.save(save_mkm_path, best_mkm_model)
        print(f"Лучшая МКМ-модель сохранена в: {save_mkm_path}")

    print("\nGA-оптимизация завершена.")
    print(f"Время поиска (только GA): {search_time_sec:.3f} сек")
    print(
        f"Поколений отработано: COLL={coll_result.generations_ran}, "
        f"GLIN={glin_result.generations_ran}"
    )
    print(f"Локальный score COLL: {coll_result.best_score:.8f}")
    print(f"Локальный score GLIN: {glin_result.best_score:.8f}")
    print(f"Итоговый Q score: {quality_score:.8f}")

    print("\nЛучшая матрица COLL:")
    print(coll_result.best_matrix)
    print("\nЛучшая матрица GLIN:")
    print(glin_result.best_matrix)

    print("\nМетрики лучшей МКМ-модели:")
    print(f"1) Доля отрицательных значений: {negative_share:.8f}")
    print(f"2) Доля глин, где сумма глин < 30%: {glin_bad_share:.8f}")
    print(f"3) Доля коллекторов, где сумма глин > 30%: {coll_bad_share:.8f}")

    print(f"\nЛучшая матрица COLL сохранена в: {best_coll_path}")
    print(f"Лучшая матрица GLIN сохранена в: {best_glin_path}")
    print(f"График лучшей МКМ-модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
