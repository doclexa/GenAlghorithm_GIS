"""Нормализация строк МКМ: на случай возникновения отрицательных значений в мкм"""

import numpy as np


def scale_pos_neg_unit_sums(v):
    """Рескейл одной строки"""
    v = np.asarray(v, dtype=float)
    w = np.zeros_like(v)

    pos = v > 0
    neg = v < 0

    if np.any(pos):
        pos_sum = float(v[pos].sum())
        if pos_sum > 0:
            w[pos] = v[pos] / pos_sum
        else:
            w[pos] = v[pos]

    if np.any(neg):
        neg_sum = float(v[neg].sum())
        if neg_sum < 0:
            w[neg] = v[neg] / (-neg_sum)
        else:
            w[neg] = v[neg]

    return w


def scale_pos_neg_unit_sums_rows(X: np.ndarray) -> np.ndarray:
    """рескейл для каждой строки матрицы (n, k)"""
    X = np.asarray(X, dtype=float, order="C")
    if X.ndim == 1:
        return scale_pos_neg_unit_sums(X)
    if X.ndim != 2:
        raise ValueError(f"Ожидается 1D или 2D массив, получено {X.ndim}D.")

    w = np.zeros_like(X)
    pos_mask = X > 0
    neg_mask = X < 0
    pos_sum = np.sum(np.where(pos_mask, X, 0.0), axis=1, keepdims=True)
    neg_sum = np.sum(np.where(neg_mask, X, 0.0), axis=1, keepdims=True)

    pos_scaled = np.zeros_like(X)
    np.divide(X, pos_sum, out=pos_scaled, where=pos_mask & (pos_sum > 0))
    w = np.where(pos_mask & (pos_sum > 0), pos_scaled, np.where(pos_mask, X, w))

    neg_scaled = np.zeros_like(X)
    np.divide(X, -neg_sum, out=neg_scaled, where=neg_mask & (neg_sum < 0))
    w = np.where(neg_mask & (neg_sum < 0), neg_scaled, np.where(neg_mask, X, w))
    return w