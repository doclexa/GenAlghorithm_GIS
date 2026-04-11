"""
Общая логика МКМ: загрузка LAS (универсально), расчёт модели, метрики качества, графики.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lasio as ls
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Три «стандартные» кривые; четвёртая подбирается автоматически, если не задана явно.
STANDARD_PROP_TRIPLE = ("POTA", "THOR", "RHOB")
FOURTH_PROP_CANDIDATES = (
    "TRNP",
    "WNKT",
    "NPHI",
    "TNPH",
    "NPOR",
    "CACO",
    "GR",
    "CNP",
)


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    base = base_dir if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


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

    missing_std = [m for m in STANDARD_PROP_TRIPLE if m not in keyset]
    if missing_std:
        raise ValueError(
            f"Автовыбор свойств: ожидаются {STANDARD_PROP_TRIPLE}. Не хватает: {missing_std}. "
            f"Доступно: {sorted(keyset)}. Задайте явно: --props M1 M2 M3 M4"
        )
    for cand in FOURTH_PROP_CANDIDATES:
        if cand in keyset:
            return (*STANDARD_PROP_TRIPLE, cand)
    raise ValueError(
        f"Не найдена четвёртая кривая из списка {FOURTH_PROP_CANDIDATES}. "
        "Укажите явно: --props M1 M2 M3 M4"
    )


def load_mkm_from_las(
    las_path: Path,
    *,
    depth_mnem: str = "DEPT",
    litho_mnem: str = "LITO",
    prop_mnems: Sequence[str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    data = np.c_[data, np.ones(data.shape[0])]
    data[data[:, 1] != 1, 1] = 2

    is_coll = data[:, 1] == 1
    is_glin = data[:, 1] == 2

    coll_prop = data[is_coll][:, 2:]
    glin_prop = data[is_glin][:, 2:]

    if verbose:
        print(f"LAS: {las_path}")
        print(f"  глубина={depth_mnem}, литология={litho_mnem}, свойства={props}")

    return data, is_coll, is_glin, coll_prop, glin_prop


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
