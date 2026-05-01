"""
Общая логика МКМ: загрузка LAS (универсально), расчёт модели, метрики качества, графики.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lasio as ls
import matplotlib.pyplot as plt
import numpy as np

from scale import scale_pos_neg_unit_sums_rows

PROJECT_ROOT = Path(__file__).resolve().parent

# Относительно PROJECT_ROOT; в CLI укажите `--las` для другой скважины.
DEFAULT_LAS_RELPATH = "data/las/621_1700_1780.las"

# Фиксированный набор четырёх кривых-свойств, если явно не задано --props.
STANDARD_PROP_QUAD = ("POTA", "THOR", "RHOB", "WNKT")


@dataclass(frozen=True)
class LithotypeInterval:
    interval_id: int
    lithotype: int
    start_idx: int
    end_idx: int
    depth_start: float
    depth_end: float
    row_indices: np.ndarray
    prop: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.row_indices))


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    base = base_dir if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


def apply_prop_rhob_weighting_to_data(data: np.ndarray) -> None:
    """
    Для МКМ: первые два столбца четырёх кривых (колонки 2 и 3 массива data: обычно POTA, THOR)
    умножаются на третий столбец свойств (колонка 4: RHOB). Последний столбец — константа 1, не трогаем.
    """
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError("Ожидается data с колонками [DEPT, LITO, 4 кривые, …]; минимум 6 столбцов до константы.")
    rh = np.asarray(data[:, 4], dtype=float)
    data[:, 2] = np.asarray(data[:, 2], dtype=float) * rh
    data[:, 3] = np.asarray(data[:, 3], dtype=float) * rh


def prepare_mkm_matrix_for_application(matrix: np.ndarray) -> np.ndarray:
    """
    Копия матрицы 5×5: первый и второй столбцы поэлементно умножаются на третий столбец (тот же смысл, что для кривых).
    Исходный массив не меняется; на диск сохраняются «сырые» матрицы, вес применяется только при расчёте.
    """
    validate_matrix_shape(matrix, "matrix")
    m = np.asarray(matrix, dtype=float, copy=True)
    w = m[:, 2]
    m[:, 0] *= w
    m[:, 1] *= w
    return m


def infer_property_mnemonics(
    las_keys: Sequence[str],
    prop_mnems: Sequence[str] | None,
) -> tuple[str, ...]:
    keyset = set(las_keys)
    if prop_mnems is not None:
        props = tuple(prop_mnems)
        if len(props) != 4:
            raise ValueError(
                "Нужно ровно 4 мнемоники кривых-свойств (после LITO идут 4 столбца + единицы → 5×5)."
            )
        missing = [m for m in props if m not in keyset]
        if missing:
            raise ValueError(
                f"В LAS нет кривых: {missing}. Доступные мнемоники: {sorted(keyset)}"
            )
        return props

    missing_std = [m for m in STANDARD_PROP_QUAD if m not in keyset]
    if missing_std:
        raise ValueError(
            f"Ожидаются кривые {STANDARD_PROP_QUAD}. Не хватает: {missing_std}. "
            f"Доступно: {sorted(keyset)}. Задайте явно: --props M1 M2 M3 M4"
        )
    return STANDARD_PROP_QUAD


def load_mkm_from_las(
    las_path: Path,
    *,
    depth_mnem: str = "DEPT",
    litho_mnem: str = "LITO",
    prop_mnems: Sequence[str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lasdata = ls.read(las_path)
    keys = list(lasdata.keys())
    keyset = set(keys)
    for req, label in ((depth_mnem, "глубины"), (litho_mnem, "литологии")):
        if req not in keyset:
            raise ValueError(
                f"Кривая {req!r} ({label}) отсутствует в LAS. Доступно: {sorted(keyset)}"
            )

    props = infer_property_mnemonics(keys, prop_mnems)
    curve_order = [depth_mnem, litho_mnem, *props]
    data = lasdata.stack_curves(curve_order, sort_curves=False)

    litho_raw = np.asarray(data[:, 1], dtype=float).copy()

    data = np.c_[data, np.ones(data.shape[0])]
    data[data[:, 1] != 1, 1] = 2

    apply_prop_rhob_weighting_to_data(data)

    is_coll = data[:, 1] == 1
    is_glin = data[:, 1] == 2

    coll_prop = data[is_coll][:, 2:]
    glin_prop = data[is_glin][:, 2:]

    if verbose:
        print(f"LAS: {las_path}")
        print(f"  глубина={depth_mnem}, литология={litho_mnem}, свойства={props}")

    return data, is_coll, is_glin, coll_prop, glin_prop, litho_raw


# Совместимость со старым именем
load_data_from_las = load_mkm_from_las


def calc_mkm_model(
    data: np.ndarray,
    is_coll: np.ndarray,
    is_glin: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    a_coll: np.ndarray,
    a_glin: np.ndarray,
) -> np.ndarray:
    a_coll_w = prepare_mkm_matrix_for_application(a_coll)
    a_glin_w = prepare_mkm_matrix_for_application(a_glin)
    inv_m_coll = np.linalg.inv(a_coll_w)
    inv_m_glin = np.linalg.inv(a_glin_w)

    mkm_coll = coll_prop @ inv_m_coll
    mkm_glin = glin_prop @ inv_m_glin

    mkm = np.zeros_like(data)
    mkm[is_coll, :] = np.hstack((data[is_coll, :2], mkm_coll))
    mkm[is_glin, :] = np.hstack((data[is_glin, :2], mkm_glin))
    return mkm


def scale_mkm_model_for_metrics(mkm_model: np.ndarray) -> np.ndarray:
    """Копия МКМ с построчным рескейлом пяти компонент (столбцы 2..6), как перед расчётом метрик."""
    out = np.asarray(mkm_model, dtype=float, copy=True)
    out[:, 2:] = scale_pos_neg_unit_sums_rows(out[:, 2:])
    return out


def split_lithotype_intervals(data: np.ndarray) -> list[LithotypeInterval]:
    litho = np.asarray(data[:, 1], dtype=int)
    depth = np.asarray(data[:, 0], dtype=float)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError("Ожидается массив data формата [depth, litho, 5 свойств].")
    if len(data) == 0:
        return []

    intervals: list[LithotypeInterval] = []
    start_idx = 0
    current_litho = int(litho[0])

    for idx in range(1, len(data) + 1):
        is_boundary = idx == len(data) or int(litho[idx]) != current_litho
        if not is_boundary:
            continue

        row_indices = np.arange(start_idx, idx, dtype=int)
        intervals.append(
            LithotypeInterval(
                interval_id=len(intervals),
                lithotype=current_litho,
                start_idx=start_idx,
                end_idx=idx,
                depth_start=float(depth[start_idx]),
                depth_end=float(depth[idx - 1]),
                row_indices=row_indices,
                prop=np.asarray(data[start_idx:idx, 2:], dtype=float),
            )
        )

        if idx < len(data):
            start_idx = idx
            current_litho = int(litho[idx])

    return intervals


def calc_mkm_model_by_intervals(
    data: np.ndarray,
    intervals: Sequence[LithotypeInterval],
    interval_matrices: dict[int, np.ndarray],
) -> np.ndarray:
    mkm = np.zeros_like(data)
    for interval in intervals:
        matrix = interval_matrices.get(interval.interval_id)
        if matrix is None:
            raise KeyError(f"Не найдена матрица для интервала {interval.interval_id}.")
        matrix_w = prepare_mkm_matrix_for_application(matrix)
        inv_matrix = np.linalg.inv(matrix_w)
        mkm_interval = interval.prop @ inv_matrix
        rows = interval.row_indices
        mkm[rows, :] = np.hstack((data[rows, :2], mkm_interval))
    return mkm


def calc_metrics_mkm(mkm_model: np.ndarray) -> tuple[float, float, float]:
    mkm_components = scale_pos_neg_unit_sums_rows(np.asarray(mkm_model[:, 2:], dtype=float))

    negative_share = np.sum(mkm_components < 0) / mkm_components.size

    is_glin = mkm_model[:, 1] == 2
    is_coll = mkm_model[:, 1] == 1

    if not np.any(is_glin):
        raise ValueError("В МКМ модели нет строк глин (LITO == 2).")
    if not np.any(is_coll):
        raise ValueError("В МКМ модели нет строк коллекторов (LITO == 1).")

    g0 = np.maximum(0.0, mkm_components[is_glin, 0])
    g1 = np.maximum(0.0, mkm_components[is_glin, 1])
    glin_sum_less_30_share = ((g0 + g1) < 0.3).sum() / np.sum(is_glin)

    c0 = np.maximum(0.0, mkm_components[is_coll, 0])
    c1 = np.maximum(0.0, mkm_components[is_coll, 1])
    coll_sum_more_30_share = ((c0 + c1) > 0.3).sum() / np.sum(is_coll)

    return (
        float(negative_share),
        float(glin_sum_less_30_share),
        float(coll_sum_more_30_share),
    )


def calc_quality_score(
    negative_share: float,
    glin_bad_share: float,
    coll_bad_share: float,
    w_negative: float,
    w_glin: float,
    w_coll: float,
) -> float:
    return (
        w_negative * negative_share
        + w_glin * glin_bad_share
        + w_coll * coll_bad_share
    )


def calc_coll_metrics(matrix: np.ndarray, coll_prop: np.ndarray) -> tuple[float, float]:
    inv_m = np.linalg.inv(prepare_mkm_matrix_for_application(matrix))
    mkm_coll = scale_pos_neg_unit_sums_rows(coll_prop @ inv_m)
    neg_share = np.sum(mkm_coll < 0) / mkm_coll.size
    c0 = np.maximum(0.0, mkm_coll[:, 0])
    c1 = np.maximum(0.0, mkm_coll[:, 1])
    coll_sum_more_30_share = ((c0 + c1) > 0.3).sum() / len(mkm_coll)
    return float(neg_share), float(coll_sum_more_30_share)


def calc_glin_metrics(matrix: np.ndarray, glin_prop: np.ndarray) -> tuple[float, float]:
    inv_m = np.linalg.inv(prepare_mkm_matrix_for_application(matrix))
    mkm_glin = scale_pos_neg_unit_sums_rows(glin_prop @ inv_m)
    neg_share = np.sum(mkm_glin < 0) / mkm_glin.size
    g0 = np.maximum(0.0, mkm_glin[:, 0])
    g1 = np.maximum(0.0, mkm_glin[:, 1])
    glin_sum_less_30_share = ((g0 + g1) < 0.3).sum() / len(mkm_glin)
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


def save_mkm_plot(
    mkm_model: np.ndarray,
    output_png_path: Path,
    *,
    litho_raw: np.ndarray | None = None,
    litho_mnem: str = "LITO",
    intervals: Sequence[LithotypeInterval] | None = None,
    summary_lines: Sequence[str] | None = None,
) -> None:
    depth = mkm_model[:, 0]

    if litho_raw is not None:
        if len(litho_raw) != len(depth):
            raise ValueError(
                f"litho_raw: ожидается длина {len(depth)}, получено {len(litho_raw)}"
            )
        ncols = 6
        fig_w = 18
    else:
        ncols = 5
        fig_w = 15

    extra_width = 2 if summary_lines else 0
    fig, axes = plt.subplots(ncols=ncols, figsize=(fig_w + extra_width, 15), sharex=False, sharey=True)

    i0 = 0
    if litho_raw is not None:
        axes[0].step(litho_raw, depth, where="mid", color="#5c4033", linewidth=1.2)
        axes[0].set_xlabel(litho_mnem)
        axes[0].set_title(f"{litho_mnem} (из LAS, без изменений)")
        axes[0].grid(True)
        i0 = 1

    plot_with_sign(axes[i0 + 0], mkm_model[:, 2], depth, "blue", "red")
    axes[i0 + 0].set_title("Глина1")

    plot_with_sign(axes[i0 + 1], mkm_model[:, 3], depth, "green", "darkred")
    axes[i0 + 1].set_title("Глина2")

    plot_with_sign(axes[i0 + 2], mkm_model[:, 4], depth, "orange", "maroon")
    axes[i0 + 2].set_title("ПШ")

    plot_with_sign(axes[i0 + 3], mkm_model[:, 5], depth, "purple", "crimson")
    axes[i0 + 3].set_title("Кварц")

    plot_with_sign(axes[i0 + 4], mkm_model[:, 6], depth, "black", "firebrick")
    axes[i0 + 4].set_title("Пористость")

    for ax in axes[i0:]:
        ax.legend()
        ax.grid(True)
        if intervals is not None:
            for interval in intervals[1:]:
                ax.axhline(interval.depth_start, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    depth_pad = max(0.5, 0.05 * float(depth.max() - depth.min()))
    for ax in axes:
        ax.set_ylim(float(depth.max() + depth_pad), float(depth.min() - depth_pad))

    fig.text(0.06, 0.5, "Глубина", va="center", rotation="vertical", fontsize=14)
    if summary_lines:
        fig.text(
            0.985,
            0.5,
            "\n".join(summary_lines),
            va="center",
            ha="right",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    plt.suptitle("Зависимости по глубине", fontsize=16)
    right_rect = 0.96 if summary_lines else 1.0
    plt.tight_layout(rect=[0, 0.03, right_rect, 0.95])

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mkm_plot_data_npz(
    output_npz_path: Path,
    mkm_model: np.ndarray,
    *,
    litho_raw: np.ndarray | None = None,
    litho_mnem: str = "LITO",
    intervals: Sequence[LithotypeInterval] | None = None,
) -> None:
    """Сохраняет числовые ряды, из которых строится `save_mkm_plot` (для постобработки / других инструментов).

    Файл .npz (compress): одна скважина — один архив.
    - mkm_model: float64, форма (N, 7) — те же данные, что на графике: [:,0] глубина, [:,1] код литотипа,
      [:,2:] пять компонент после `scale_mkm_model_for_metrics` (Глина1 … Пористость).
    - litho_las: опционально, сырой LITO из LAS, длина N (если был передан litho_raw).
    - interval_axhline_depth: глубины горизонтальных линий между интервалами (как в `save_mkm_plot`, без первого).
    - meta_json_utf8: JSON с полями version, litho_mnem, mkm_column_keys (имена столбцов mkm_model).

    Чтение: ``d = np.load(path); meta = json.loads(d['meta_json_utf8'].tobytes().decode('utf-8'))``
    """
    mkm_f = np.asarray(mkm_model, dtype=np.float64)
    if mkm_f.ndim != 2 or mkm_f.shape[1] != 7:
        raise ValueError(f"mkm_model ожидается (N, 7), сейчас: {mkm_f.shape}")

    meta = {
        "version": 1,
        "litho_mnem": litho_mnem,
        "mkm_column_keys": [
            "depth",
            "lithotype_code",
            "glin1",
            "glin2",
            "psh",
            "quartz",
            "porosity",
        ],
    }
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    meta_arr = np.frombuffer(meta_bytes, dtype=np.uint8)

    if intervals is not None:
        axh = np.array([float(iv.depth_start) for iv in intervals[1:]], dtype=np.float64)
    else:
        axh = np.zeros((0,), dtype=np.float64)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, np.ndarray] = {
        "mkm_model": mkm_f,
        "interval_axhline_depth": axh,
        "meta_json_utf8": meta_arr,
    }
    if litho_raw is not None:
        lr = np.asarray(litho_raw)
        if lr.shape[0] != mkm_f.shape[0]:
            raise ValueError(
                f"litho_raw: длина {lr.shape[0]}, для mkm_model ожидается {mkm_f.shape[0]}"
            )
        save_kw["litho_las"] = lr.astype(np.float64, copy=False)
    np.savez_compressed(output_npz_path, **save_kw)


def validate_matrix_shape(matrix: np.ndarray, matrix_name: str) -> None:
    if matrix.shape != (5, 5):
        raise ValueError(f"{matrix_name} должна иметь размер 5x5, сейчас: {matrix.shape}")


def flatten_bounds(a_min: np.ndarray, a_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_min = a_min.flatten().astype(float)
    flat_max = a_max.flatten().astype(float)
    return flat_min, flat_max


def validate_k_shape(k_matrix: np.ndarray, matrix_name: str) -> None:
    if k_matrix.shape != (5, 5):
        raise ValueError(f"{matrix_name} должна иметь размер 5x5, сейчас: {k_matrix.shape}")
    if np.any(k_matrix < 1):
        raise ValueError(f"{matrix_name} содержит значения < 1, это недопустимо.")


def default_mkm_artifact_paths(
    project_root: Path,
    las_stem: str,
    method_suffix: str,
) -> dict[str, Path]:
    """Стандартные пути: outputs/plots, outputs/matrices."""
    return {
        "plot": project_root / "outputs" / "plots" / f"{las_stem}_mkm_{method_suffix}.png",
        "coll": project_root / "outputs" / "matrices" / f"{las_stem}_coll_{method_suffix}.out",
        "glin": project_root / "outputs" / "matrices" / f"{las_stem}_glin_{method_suffix}.out",
    }
