"""МКМ по заданным лабораторным матрицам (без оптимизации)."""

from __future__ import annotations

import argparse

import numpy as np

from mkm_core import (
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    calc_mkm_model,
    calc_metrics_mkm,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    validate_matrix_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Считает МКМ по фиксированным матрицам matrix_coll / matrix_glina, "
            "три метрики качества и график по пяти компонентам."
        )
    )
    parser.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las (по умолчанию основная скважина проекта; иначе укажите явно).",
    )
    parser.add_argument("--depth", default="DEPT", help="Мнемоника глубины.")
    parser.add_argument("--litho", default="LITO", help="Мнемоника литологии.")
    parser.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства (иначе автоподбор).",
    )
    parser.add_argument(
        "--a-coll",
        default="config/matrix_coll.out",
        help="Матрица коллектора 5×5.",
    )
    parser.add_argument(
        "--a-glin",
        default="config/matrix_glina.out",
        help="Матрица глины 5×5.",
    )
    parser.add_argument("--save-mkm", default="", help="Опционально: путь для .npy МКМ.")
    parser.add_argument(
        "--plot-png",
        default="outputs/plots/mkm_lab_model_plot.png",
        help="График МКМ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    las_path = resolve_path(args.las, PROJECT_ROOT)
    a_coll_path = resolve_path(args.a_coll, PROJECT_ROOT)
    a_glin_path = resolve_path(args.a_glin, PROJECT_ROOT)

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

    a_coll = np.loadtxt(a_coll_path)
    a_glin = np.loadtxt(a_glin_path)

    validate_matrix_shape(a_coll, "A_coll")
    validate_matrix_shape(a_glin, "A_glin")

    mkm_model = calc_mkm_model(data, is_coll, is_glin, coll_prop, glin_prop, a_coll, a_glin)

    negative_share, glin_sum_less_30_share, coll_sum_more_30_share = calc_metrics_mkm(mkm_model)

    print("Рассчитаны метрики по всей МКМ модели:")
    print(f"1) Доля отрицательных значений: {negative_share:.6f}")
    print(f"2) Доля глин, где сумма глин < 30%: {glin_sum_less_30_share:.6f}")
    print(f"3) Доля коллекторов, где сумма глин > 30%: {coll_sum_more_30_share:.6f}")

    if args.save_mkm:
        save_path = resolve_path(args.save_mkm, PROJECT_ROOT)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, mkm_model)
        print(f"МКМ модель сохранена в: {save_path}")

    plot_path = resolve_path(args.plot_png, PROJECT_ROOT)
    save_mkm_plot(
        mkm_model,
        plot_path,
        litho_raw=litho_raw,
        litho_mnem=args.litho,
    )
    print(f"График МКМ модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
