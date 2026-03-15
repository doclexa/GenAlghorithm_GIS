from __future__ import annotations

import argparse
import itertools
import time
from math import prod
from pathlib import Path

import lasio as ls
import matplotlib.pyplot as plt
import numpy as np


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_data_from_las(
    las_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return (
        float(negative_share),
        float(glin_sum_less_30_share),
        float(coll_sum_more_30_share),
    )


def calc_coll_metrics(matrix: np.ndarray, coll_prop: np.ndarray) -> tuple[float, float]:
    inv_m = np.linalg.inv(matrix)
    mkm_coll = coll_prop @ inv_m
    neg_share = np.sum(mkm_coll < 0) / mkm_coll.size
    coll_sum_more_30_share = (mkm_coll[:, 0] + mkm_coll[:, 1] > 0.3).sum() / len(mkm_coll)
    return float(neg_share), float(coll_sum_more_30_share)


def calc_glin_metrics(matrix: np.ndarray, glin_prop: np.ndarray) -> tuple[float, float]:
    inv_m = np.linalg.inv(matrix)
    mkm_glin = glin_prop @ inv_m
    neg_share = np.sum(mkm_glin < 0) / mkm_glin.size
    glin_sum_less_30_share = (mkm_glin[:, 0] + mkm_glin[:, 1] < 0.3).sum() / len(mkm_glin)
    return float(neg_share), float(glin_sum_less_30_share)


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


def validate_k_shape(k_matrix: np.ndarray, matrix_name: str) -> None:
    if k_matrix.shape != (5, 5):
        raise ValueError(f"{matrix_name} должна иметь размер 5x5, сейчас: {k_matrix.shape}")
    if np.any(k_matrix < 1):
        raise ValueError(f"{matrix_name} содержит значения < 1, это недопустимо.")


def build_value_grids(
    a_min: np.ndarray,
    a_max: np.ndarray,
    a_k: np.ndarray,
) -> list[np.ndarray]:
    grids: list[np.ndarray] = []
    flat_min = a_min.flatten()
    flat_max = a_max.flatten()
    flat_k = a_k.flatten()

    for min_val, max_val, k_raw in zip(flat_min, flat_max, flat_k):
        k = int(round(float(k_raw)))
        if k <= 1 or np.isclose(min_val, max_val):
            values = np.array([float(min_val)], dtype=float)
        else:
            values = np.linspace(float(min_val), float(max_val), num=k, dtype=float)
        grids.append(values)

    return grids


def matrix_generator(grids: list[np.ndarray]):
    for values in itertools.product(*grids):
        yield np.array(values, dtype=float).reshape(5, 5)


def matrix_to_inline_params(matrix: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in matrix.flatten())


def brute_force_best_coll(
    coll_prop: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    a_k: np.ndarray,
    neg_weight_scaled: float,
    coll_weight: float,
    global_start_index: int,
    global_total: int,
    max_iterations: int | None,
) -> tuple[np.ndarray, float, float, float, int, int]:
    grids = build_value_grids(a_min, a_max, a_k)
    total = prod(len(g) for g in grids)
    if max_iterations is not None:
        total = min(total, max_iterations)

    best_score = float("inf")
    best_matrix: np.ndarray | None = None
    best_neg = float("inf")
    best_coll_bad = float("inf")
    invalid_count = 0

    for local_iter, matrix in enumerate(matrix_generator(grids), start=1):
        if max_iterations is not None and local_iter > max_iterations:
            break

        global_iter = global_start_index + local_iter
        try:
            neg_share, coll_bad_share = calc_coll_metrics(matrix, coll_prop)
            score = neg_weight_scaled * neg_share + coll_weight * coll_bad_share

            print(
                f"[Итерация {global_iter}/{global_total}] [COLL] "
                f"score={score:.8f} neg={neg_share:.8f} coll_bad={coll_bad_share:.8f} "
                f"params={matrix_to_inline_params(matrix)}"
            )

            if score < best_score:
                best_score = score
                best_matrix = matrix.copy()
                best_neg = neg_share
                best_coll_bad = coll_bad_share
        except np.linalg.LinAlgError:
            invalid_count += 1
            print(
                f"[Итерация {global_iter}/{global_total}] [COLL] "
                f"matrix_singular=True params={matrix_to_inline_params(matrix)}"
            )

    if best_matrix is None:
        raise RuntimeError("Для COLL не найдено ни одной обратимой матрицы.")

    return best_matrix, best_score, best_neg, best_coll_bad, total, invalid_count


def brute_force_best_glin(
    glin_prop: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    a_k: np.ndarray,
    neg_weight_scaled: float,
    glin_weight: float,
    global_start_index: int,
    global_total: int,
    max_iterations: int | None,
) -> tuple[np.ndarray, float, float, float, int, int]:
    grids = build_value_grids(a_min, a_max, a_k)
    total = prod(len(g) for g in grids)
    if max_iterations is not None:
        total = min(total, max_iterations)

    best_score = float("inf")
    best_matrix: np.ndarray | None = None
    best_neg = float("inf")
    best_glin_bad = float("inf")
    invalid_count = 0

    for local_iter, matrix in enumerate(matrix_generator(grids), start=1):
        if max_iterations is not None and local_iter > max_iterations:
            break

        global_iter = global_start_index + local_iter
        try:
            neg_share, glin_bad_share = calc_glin_metrics(matrix, glin_prop)
            score = neg_weight_scaled * neg_share + glin_weight * glin_bad_share

            print(
                f"[Итерация {global_iter}/{global_total}] [GLIN] "
                f"score={score:.8f} neg={neg_share:.8f} glin_bad={glin_bad_share:.8f} "
                f"params={matrix_to_inline_params(matrix)}"
            )

            if score < best_score:
                best_score = score
                best_matrix = matrix.copy()
                best_neg = neg_share
                best_glin_bad = glin_bad_share
        except np.linalg.LinAlgError:
            invalid_count += 1
            print(
                f"[Итерация {global_iter}/{global_total}] [GLIN] "
                f"matrix_singular=True params={matrix_to_inline_params(matrix)}"
            )

    if best_matrix is None:
        raise RuntimeError("Для GLIN не найдено ни одной обратимой матрицы.")

    return best_matrix, best_score, best_neg, best_glin_bad, total, invalid_count


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Брутфорс по дискретным значениям матриц A_coll/A_glin, расчет лучших матриц, "
            "метрик МКМ и построение графика по лучшему варианту."
        )
    )
    parser.add_argument("--las", default="inp.las", help="Путь к .las файлу.")
    parser.add_argument("--a-min-coll", default="a_min_coll.in", help="Путь к a_min_coll.in.")
    parser.add_argument("--a-max-coll", default="a_max_coll.in", help="Путь к a_max_coll.in.")
    parser.add_argument("--a-k-coll", default="a_k_coll.in", help="Путь к a_k_coll.in.")
    parser.add_argument("--a-min-glin", default="a_min_glin.in", help="Путь к a_min_glin.in.")
    parser.add_argument("--a-max-glin", default="a_max_glin.in", help="Путь к a_max_glin.in.")
    parser.add_argument("--a-k-glin", default="a_k_glin.in", help="Путь к a_k_glin.in.")
    parser.add_argument(
        "--best-coll-out",
        default="best_matrix_coll_bruteforce.out",
        help="Куда сохранить лучшую матрицу коллектора.",
    )
    parser.add_argument(
        "--best-glin-out",
        default="best_matrix_glin_bruteforce.out",
        help="Куда сохранить лучшую матрицу глины.",
    )
    parser.add_argument(
        "--plot-png",
        default="mkm_bruteforce_best_plot.png",
        help="Путь для сохранения графика лучшей МКМ-модели в .png.",
    )
    parser.add_argument(
        "--save-mkm",
        default="",
        help="Необязательно: путь для сохранения лучшей МКМ-модели в .npy.",
    )
    parser.add_argument(
        "--w-negative",
        type=float,
        default=1.0,
        help="Вес метрики: доля отрицательных значений.",
    )
    parser.add_argument(
        "--w-glin",
        type=float,
        default=1.0,
        help="Вес метрики: доля глин с суммой глин < 30%.",
    )
    parser.add_argument(
        "--w-coll",
        type=float,
        default=1.0,
        help="Вес метрики: доля коллекторов с суммой глин > 30%.",
    )
    parser.add_argument(
        "--max-iter-coll",
        type=int,
        default=0,
        help="Ограничить число итераций для COLL (0 = без ограничения).",
    )
    parser.add_argument(
        "--max-iter-glin",
        type=int,
        default=0,
        help="Ограничить число итераций для GLIN (0 = без ограничения).",
    )
    args = parser.parse_args()

    las_path = resolve_path(args.las, script_dir)
    a_min_coll_path = resolve_path(args.a_min_coll, script_dir)
    a_max_coll_path = resolve_path(args.a_max_coll, script_dir)
    a_k_coll_path = resolve_path(args.a_k_coll, script_dir)
    a_min_glin_path = resolve_path(args.a_min_glin, script_dir)
    a_max_glin_path = resolve_path(args.a_max_glin, script_dir)
    a_k_glin_path = resolve_path(args.a_k_glin, script_dir)

    data, is_coll, is_glin, coll_prop, glin_prop = load_data_from_las(las_path)
    if len(coll_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 1 (коллекторы).")
    if len(glin_prop) == 0:
        raise ValueError("В данных нет строк с LITO != 1 (глины после преобразования в 2).")

    a_min_coll = np.loadtxt(a_min_coll_path)
    a_max_coll = np.loadtxt(a_max_coll_path)
    a_k_coll = np.loadtxt(a_k_coll_path)
    a_min_glin = np.loadtxt(a_min_glin_path)
    a_max_glin = np.loadtxt(a_max_glin_path)
    a_k_glin = np.loadtxt(a_k_glin_path)

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

    best_coll, best_coll_score, best_coll_neg, best_coll_bad, coll_iters, coll_invalid = brute_force_best_coll(
        coll_prop=coll_prop,
        a_min=a_min_coll,
        a_max=a_max_coll,
        a_k=a_k_coll,
        neg_weight_scaled=coll_neg_weight_scaled,
        coll_weight=args.w_coll,
        global_start_index=0,
        global_total=total_iterations,
        max_iterations=max_iter_coll,
    )

    best_glin, best_glin_score, best_glin_neg, best_glin_bad, glin_iters, glin_invalid = brute_force_best_glin(
        glin_prop=glin_prop,
        a_min=a_min_glin,
        a_max=a_max_glin,
        a_k=a_k_glin,
        neg_weight_scaled=glin_neg_weight_scaled,
        glin_weight=args.w_glin,
        global_start_index=coll_iters,
        global_total=total_iterations,
        max_iterations=max_iter_glin,
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

    best_coll_path = resolve_path(args.best_coll_out, script_dir)
    best_glin_path = resolve_path(args.best_glin_out, script_dir)
    best_coll_path.parent.mkdir(parents=True, exist_ok=True)
    best_glin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(best_coll_path, best_coll, fmt="%.12g")
    np.savetxt(best_glin_path, best_glin, fmt="%.12g")

    plot_path = resolve_path(args.plot_png, script_dir)
    save_mkm_plot(best_mkm_model, plot_path)

    if args.save_mkm:
        save_mkm_path = resolve_path(args.save_mkm, script_dir)
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
