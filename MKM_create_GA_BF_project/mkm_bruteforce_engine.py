"""Полный перебор матриц A_coll / A_glin по сетке из a_k_*.in."""

from __future__ import annotations

import itertools
from math import prod

import numpy as np

from mkm_core import calc_coll_metrics, calc_glin_metrics


def build_value_grids(
    a_min: np.ndarray,
    a_max: np.ndarray,
    a_k: np.ndarray,
) -> list[np.ndarray]:
    """Готовит сетку допустимых значений для каждого из 25 параметров матрицы."""
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
    """Генератор всех комбинаций параметров в виде матриц 5x5."""
    for values in itertools.product(*grids):
        yield np.array(values, dtype=float).reshape(5, 5)


def matrix_to_inline_params(matrix: np.ndarray) -> str:
    """Упаковывает матрицу в строку параметров для логов прогресса."""
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
    verbose: bool = True,
) -> tuple[np.ndarray, float, float, float, int, int]:
    """Ищет лучшую матрицу для коллектора полным перебором по заданной сетке."""
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

            if verbose:
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
            if verbose:
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
    verbose: bool = True,
) -> tuple[np.ndarray, float, float, float, int, int]:
    """Ищет лучшую матрицу для глин полным перебором по заданной сетке."""
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

            if verbose:
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
            if verbose:
                print(
                    f"[Итерация {global_iter}/{global_total}] [GLIN] "
                    f"matrix_singular=True params={matrix_to_inline_params(matrix)}"
                )

    if best_matrix is None:
        raise RuntimeError("Для GLIN не найдено ни одной обратимой матрицы.")

    return best_matrix, best_score, best_neg, best_glin_bad, total, invalid_count
