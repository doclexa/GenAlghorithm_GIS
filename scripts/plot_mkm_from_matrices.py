#!/usr/bin/env python3
"""
Две матрицы (COLL, GLIN) + LAS → график МКМ по глубине (6 колонок: исходный LITO + 5 компонент).

Ось глубины: меньшие значения сверху, большие вниз (set_ylim без invert_yaxis).

Запуск из корня GenAlghorithm_GIS:
  python scripts/plot_mkm_from_matrices.py
  python scripts/plot_mkm_from_matrices.py --las data/las/skv621.las \\
    --matrix-coll outputs/matrices/matrix_coll_answer.out \\
    --matrix-glin outputs/matrices/matrix_glina_answer.out \\
    --output outputs/plots/mkm_answer_depth.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    calc_mkm_model,
    load_mkm_from_las,
    plot_with_sign,
    resolve_path,
    scale_mkm_model_for_metrics,
    validate_matrix_shape,
)


def plot_mkm_depth_figure(
    mkm_model: np.ndarray,
    litho_raw: np.ndarray,
    *,
    litho_mnem: str,
    output_png_path: Path,
    title: str = "МКМ по глубине (заданные матрицы)",
) -> None:
    depth = mkm_model[:, 0]
    if len(litho_raw) != len(depth):
        raise ValueError(
            f"LITO: длина {len(litho_raw)} не совпадает с числом точек по стволу {len(depth)}"
        )

    ncols = 6
    fig, axes = plt.subplots(ncols=ncols, figsize=(18, 15), sharex=False, sharey=True)

    axes[0].step(litho_raw, depth, where="mid", color="#5c4033", linewidth=1.2)
    axes[0].set_xlabel(litho_mnem)
    axes[0].set_title(f"{litho_mnem} (из LAS, без изменений)")
    axes[0].grid(True)

    plot_with_sign(axes[1], mkm_model[:, 2], depth, "blue", "red")
    axes[1].set_title("Глина1")

    plot_with_sign(axes[2], mkm_model[:, 3], depth, "green", "darkred")
    axes[2].set_title("Глина2")

    plot_with_sign(axes[3], mkm_model[:, 4], depth, "orange", "maroon")
    axes[3].set_title("ПШ")

    plot_with_sign(axes[4], mkm_model[:, 5], depth, "purple", "crimson")
    axes[4].set_title("Кварц")

    plot_with_sign(axes[5], mkm_model[:, 6], depth, "black", "firebrick")
    axes[5].set_title("Пористость")

    for ax in axes[1:]:
        ax.legend()
        ax.grid(True)

    dmin = float(np.nanmin(depth))
    dmax = float(np.nanmax(depth))
    if dmax < dmin:
        dmin, dmax = dmax, dmin
    for ax in axes:
        ax.set_ylim(dmax, dmin)

    fig.text(0.06, 0.5, "Глубина", va="center", rotation="vertical", fontsize=14)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LAS + две матрицы 5×5 → PNG МКМ по глубине.")
    p.add_argument("--project-root", default="", help="Корень проекта (пусто = рядом с mkm_core.py).")
    p.add_argument("--las", default=DEFAULT_LAS_RELPATH, help="Путь к .las.")
    p.add_argument(
        "--matrix-coll",
        default="outputs/matrices/matrix_coll_answer.out",
        help="Матрица коллектора 5×5.",
    )
    p.add_argument(
        "--matrix-glin",
        default="outputs/matrices/matrix_glina_answer.out",
        help="Матрица глины 5×5.",
    )
    p.add_argument(
        "--output",
        default="outputs/plots/mkm_from_matrices_depth.png",
        help="Куда сохранить PNG.",
    )
    p.add_argument("--depth", default="DEPT", help="Мнемоника глубины.")
    p.add_argument("--litho", default="LITO", help="Мнемоника литологии в LAS.")
    p.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства (иначе автоподбор как в mkm_run_ga).",
    )
    p.add_argument("--title", default="", help="Заголовок рисунка (пусто = по умолчанию).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT

    las_path = resolve_path(args.las, root)
    coll_path = resolve_path(args.matrix_coll, root)
    glin_path = resolve_path(args.matrix_glin, root)
    out_path = resolve_path(args.output, root)

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
        raise ValueError("В данных нет строк с LITO == 2 (глины после нормализации).")

    def load_matrix(path: Path) -> np.ndarray:
        try:
            return np.loadtxt(path, comments=("M", "#"))
        except ValueError:
            return np.loadtxt(path, skiprows=1)

    a_coll = load_matrix(coll_path)
    a_glin = load_matrix(glin_path)
    validate_matrix_shape(a_coll, "matrix_coll")
    validate_matrix_shape(a_glin, "matrix_glin")

    mkm = calc_mkm_model(data, is_coll, is_glin, coll_prop, glin_prop, a_coll, a_glin)
    mkm = scale_mkm_model_for_metrics(mkm)

    title = args.title.strip() or f"МКМ по глубине — {las_path.name}"
    plot_mkm_depth_figure(
        mkm,
        litho_raw,
        litho_mnem=args.litho,
        output_png_path=out_path,
        title=title,
    )
    print(f"График сохранён: {out_path}")


if __name__ == "__main__":
    main()
