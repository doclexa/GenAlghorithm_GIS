from __future__ import annotations

import argparse
from pathlib import Path

import lasio as ls
import matplotlib.pyplot as plt
import numpy as np


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_data_from_las(las_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lasdata = ls.read(las_path)

    data = lasdata.stack_curves(
        ["DEPT", "LITO", "POTA", "THOR", "RHOB", "TRNP"],
        sort_curves=False,
    )

    data = np.c_[data, np.ones(data.shape[0])]
    data[data[:, 1] != 1, 1] = 2

    is_coll = data[:, 1] == 1
    is_glin = data[:, 1] == 2

    coll_prop = data[is_coll][:, 2:]
    glin_prop = data[is_glin][:, 2:]

    return data, is_coll, is_glin, coll_prop, glin_prop


def calc_mkm_model(
    data: np.ndarray,
    is_coll: np.ndarray,
    is_glin: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    a_coll: np.ndarray,
    a_glin: np.ndarray,
) -> np.ndarray:
    inv_m_coll = np.linalg.inv(a_coll)
    inv_m_glin = np.linalg.inv(a_glin)

    mkm_coll = coll_prop @ inv_m_coll
    mkm_glin = glin_prop @ inv_m_glin

    mkm = np.zeros_like(data)
    mkm[is_coll, :] = np.hstack((data[is_coll, :2], mkm_coll))
    mkm[is_glin, :] = np.hstack((data[is_glin, :2], mkm_glin))
    return mkm


def calc_metrics_mkm(mkm_model: np.ndarray) -> tuple[float, float, float]:
    mkm_components = mkm_model[:, 2:]

    negative_share = np.sum(mkm_components < 0) / mkm_components.size

    is_glin = mkm_model[:, 1] == 2
    is_coll = mkm_model[:, 1] == 1

    if not np.any(is_glin):
        raise ValueError("В МКМ модели нет строк глин (LITO == 2).")
    if not np.any(is_coll):
        raise ValueError("В МКМ модели нет строк коллекторов (LITO == 1).")

    glin_sum_less_30_share = (
        mkm_components[is_glin, 0] + mkm_components[is_glin, 1] < 0.3
    ).sum() / np.sum(is_glin)
    coll_sum_more_30_share = (
        mkm_components[is_coll, 0] + mkm_components[is_coll, 1] > 0.3
    ).sum() / np.sum(is_coll)

    return float(negative_share), float(glin_sum_less_30_share), float(coll_sum_more_30_share)


def plot_with_sign(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    positive_color: str,
    negative_color: str,
) -> None:
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    ax.plot(x[pos_mask], y[pos_mask], color=positive_color, label="+")
    ax.plot(
        x[neg_mask],
        y[neg_mask],
        color=negative_color,
        label="-",
        marker="o",
        markersize=2,
        ls="",
    )


def save_mkm_plot(mkm_model: np.ndarray, output_png_path: Path) -> None:
    depth = mkm_model[:, 0]

    fig, axes = plt.subplots(ncols=5, figsize=(15, 15), sharex=False, sharey=True)

    for ax in axes:
        ax.invert_yaxis()

    plot_with_sign(axes[0], mkm_model[:, 2], depth, "blue", "red")
    axes[0].set_title("Глина1")

    plot_with_sign(axes[1], mkm_model[:, 3], depth, "green", "darkred")
    axes[1].set_title("Глина2")

    plot_with_sign(axes[2], mkm_model[:, 4], depth, "orange", "maroon")
    axes[2].set_title("ПШ")

    plot_with_sign(axes[3], mkm_model[:, 5], depth, "purple", "crimson")
    axes[3].set_title("Кварц")

    plot_with_sign(axes[4], mkm_model[:, 6], depth, "black", "firebrick")
    axes[4].set_title("Пористость")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    fig.text(0.06, 0.5, "Глубина", va="center", rotation="vertical", fontsize=14)
    plt.suptitle("Зависимости по глубине", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_matrix_shape(matrix: np.ndarray, matrix_name: str) -> None:
    if matrix.shape != (5, 5):
        raise ValueError(f"{matrix_name} должна иметь размер 5x5, сейчас: {matrix.shape}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Парсит .las как в ноутбуке, считает МКМ модель по лабораторным матрицам "
            "и выводит 3 метрики по всей модели, а также строит график по 5 кривым."
        )
    )
    parser.add_argument("--las", default="inp.las", help="Путь к .las файлу.")
    parser.add_argument("--a-coll", default="matrix_coll.out", help="Путь к matrix_coll.out.")
    parser.add_argument("--a-glin", default="matrix_glina.out", help="Путь к matrix_glina.out.")
    parser.add_argument(
        "--save-mkm",
        default="",
        help="Необязательно: путь для сохранения рассчитанной МКМ модели в .npy.",
    )
    parser.add_argument(
        "--plot-png",
        default="mkm_model_plot.png",
        help="Путь для сохранения графика МКМ-модели в .png.",
    )
    args = parser.parse_args()

    las_path = resolve_path(args.las, script_dir)
    a_coll_path = resolve_path(args.a_coll, script_dir)
    a_glin_path = resolve_path(args.a_glin, script_dir)

    data, is_coll, is_glin, coll_prop, glin_prop = load_data_from_las(las_path)

    if len(coll_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 1 (коллекторы).")
    if len(glin_prop) == 0:
        raise ValueError("В данных нет строк с LITO != 1 (глины после преобразования в 2).")

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
        save_path = resolve_path(args.save_mkm, script_dir)
        np.save(save_path, mkm_model)
        print(f"МКМ модель сохранена в: {save_path}")

    plot_path = resolve_path(args.plot_png, script_dir)
    save_mkm_plot(mkm_model, plot_path)
    print(f"График МКМ модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
