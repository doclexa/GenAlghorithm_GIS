#!/usr/bin/env python3
"""
Постобработка GA-результатов МКМ в финальную литологическую колонку по глубине.

Источник:
  outputs/mkm_result/*/*_mkm_ga_plot_data.npz

Выход:
  outputs/final_lithology/<well_name>/
    - <well_name>_final_lithology_points.csv
    - <well_name>_final_lithology_intervals.csv
    - <well_name>_final_lithology_column.png

  outputs/final_lithology/final_lithology_summary.csv
  outputs/final_lithology/final_lithology_overview.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mkm_core import PROJECT_ROOT, resolve_path  # noqa: E402

EPS = 1e-12


@dataclass(frozen=True)
class LithoStyle:
    """Визуальный стиль класса породы для построения колонок."""
    label: str
    color: str
    hatch: str


STYLES: dict[str, LithoStyle] = {
    "sand_quartz_arenite": LithoStyle("Кварцевый аренит", "#f7e27b", ".."),
    "sand_subarkose_arenite": LithoStyle("Субаркозовый аренит", "#f4c36b", "//"),
    "sand_arkose_arenite": LithoStyle("Аркозовый аренит", "#ee9f6a", "\\\\"),
    "sand_quartz_wacke": LithoStyle("Кварцевый песчаник", "#c8a35e", "xx"),
    "sand_subarkose_wacke": LithoStyle("Субаркозовый песчаник", "#b88955", "++"),
    "sand_arkose_wacke": LithoStyle("Аркозовый песчаник", "#a8704b", "oo"),
    "sand_strongly_clayey": LithoStyle("Сильно глинистый песчаник", "#8c6a4b", "OO"),
    "seal_kaolinite_clay": LithoStyle("Каолинитовая тяжелая глина", "#cbb6e9", "//"),
    "seal_hydromica_clay": LithoStyle("Гидрослюдистая тяжелая глина", "#a9bea7", "\\\\"),
    "seal_mixed_clay": LithoStyle("Каолинит-гидрослюдистая тяжелая глина", "#9ab0cf", "xx"),
    "seal_kaolinite_argillite": LithoStyle("Каолинитовая легкая глина", "#b8a4da", "//"),
    "seal_hydromica_argillite": LithoStyle("Гидрослюдистая легкая глина", "#8fa891", "\\\\"),
    "seal_mixed_argillite": LithoStyle("Каолинит-гидрослюдистая легкая глина", "#869bbb", "xx"),
    "seal_silty_argillite": LithoStyle("Песчанистая легкая глина", "#b2bcc8", "--"),
    "undefined": LithoStyle("Неопределенная порода", "#d9d9d9", ""),
}


@dataclass(frozen=True)
class ClassifiedPoint:
    """Классификация одной точки по глубине после МКМ."""
    depth: float
    lithotype_code: int
    kaolinite: float
    hydromica: float
    feldspar: float
    quartz: float
    porosity: float
    clay_share: float
    q_over_qf: float
    porosity_tag: str
    class_key: str
    class_name: str


@dataclass(frozen=True)
class ClassifiedInterval:
    """Сжатый интервал соседних точек одного и того же итогового класса."""
    start_depth: float
    end_depth: float
    thickness: float
    lithotype_code: int
    class_key: str
    class_name: str
    sample_count: int


def porosity_modifier(porosity: float) -> str:
    """Текстовая градация пористости для человекочитаемого названия класса."""
    if porosity >= 0.18:
        return "высокопористый"
    if porosity >= 0.08:
        return "пористый"
    return "плотный"


def classify_reservoir(clay_share: float, q_over_qf: float, porosity_tag: str) -> tuple[str, str]:
    """Классифицирует коллектор по глинистости и отношению кварца к кварц+полевой шпат."""
    if clay_share > 0.50:
        return "sand_strongly_clayey", f"{porosity_tag} сильно глинистый песчаник"

    if clay_share >= 0.15:
        if q_over_qf >= 0.90:
            return "sand_quartz_wacke", f"{porosity_tag} кварцевый песчаник"
        if q_over_qf >= 0.75:
            return "sand_subarkose_wacke", f"{porosity_tag} субаркозовый песчаник"
        return "sand_arkose_wacke", f"{porosity_tag} аркозовый песчаник"

    if q_over_qf >= 0.90:
        return "sand_quartz_arenite", f"{porosity_tag} кварцевый аренит"
    if q_over_qf >= 0.75:
        return "sand_subarkose_arenite", f"{porosity_tag} субаркозовый аренит"
    return "sand_arkose_arenite", f"{porosity_tag} аркозовый аренит"


def classify_seal(clay_share: float, kaolinite: float, hydromica: float, feldspar: float, quartz: float) -> tuple[str, str]:
    """Классифицирует покрышку по доле глины и типу глинистого минерала."""
    clay_total = kaolinite + hydromica
    if clay_total <= EPS and clay_share < 0.05:
        return "undefined", "неопределенная покрышечная порода"

    if clay_total > EPS:
        kaolinite_ratio = kaolinite / clay_total
    else:
        kaolinite_ratio = 0.5

    if kaolinite_ratio >= 0.65:
        clay_prefix = "каолинитовая"
        key_prefix = "kaolinite"
    elif kaolinite_ratio <= 0.35:
        clay_prefix = "гидрослюдистая"
        key_prefix = "hydromica"
    else:
        clay_prefix = "каолинит-гидрослюдистая"
        key_prefix = "mixed"

    if clay_share >= 0.67:
        return f"seal_{key_prefix}_clay", f"{clay_prefix} тяжелая глина"
    if clay_share >= 0.33:
        return f"seal_{key_prefix}_argillite", f"{clay_prefix} легкая глина"

    text_name = "песчанистая легкая глина"
    return "seal_silty_argillite", text_name


def classify_row(row: np.ndarray) -> ClassifiedPoint:
    """Классифицирует одну строку `mkm_model` в финальный литологический класс."""
    depth = float(row[0])
    lithotype_code = 1 if int(round(float(row[1]))) == 1 else 2

    kaolinite = max(float(row[2]), 0.0)
    hydromica = max(float(row[3]), 0.0)
    feldspar = max(float(row[4]), 0.0)
    quartz = max(float(row[5]), 0.0)
    porosity = max(float(row[6]), 0.0)

    solid = kaolinite + hydromica + feldspar + quartz
    clay_total = kaolinite + hydromica
    clay_share = clay_total / solid if solid > EPS else 0.0
    qf = quartz + feldspar
    q_over_qf = quartz / qf if qf > EPS else 0.0
    por_tag = porosity_modifier(porosity)

    if solid <= EPS:
        return ClassifiedPoint(
            depth=depth,
            lithotype_code=lithotype_code,
            kaolinite=kaolinite,
            hydromica=hydromica,
            feldspar=feldspar,
            quartz=quartz,
            porosity=porosity,
            clay_share=clay_share,
            q_over_qf=q_over_qf,
            porosity_tag=por_tag,
            class_key="undefined",
            class_name="неопределенная порода",
        )

    # Для коллектора и покрышки применяются разные геологические правила классификации.
    if lithotype_code == 1:
        class_key, class_name = classify_reservoir(clay_share, q_over_qf, por_tag)
    else:
        class_key, class_name = classify_seal(clay_share, kaolinite, hydromica, feldspar, quartz)

    return ClassifiedPoint(
        depth=depth,
        lithotype_code=lithotype_code,
        kaolinite=kaolinite,
        hydromica=hydromica,
        feldspar=feldspar,
        quartz=quartz,
        porosity=porosity,
        clay_share=clay_share,
        q_over_qf=q_over_qf,
        porosity_tag=por_tag,
        class_key=class_key,
        class_name=class_name,
    )


def build_depth_edges(depth: np.ndarray) -> np.ndarray:
    """Строит границы ячеек по глубине, чтобы корректно рисовать интервалы-столбики."""
    depth_f = np.asarray(depth, dtype=float)
    if depth_f.size == 0:
        return np.zeros((0,), dtype=float)
    if depth_f.size == 1:
        return np.array([depth_f[0] - 0.1, depth_f[0] + 0.1], dtype=float)

    edges = np.empty((depth_f.size + 1,), dtype=float)
    edges[1:-1] = (depth_f[:-1] + depth_f[1:]) / 2.0
    first_step = depth_f[1] - depth_f[0]
    last_step = depth_f[-1] - depth_f[-2]
    edges[0] = depth_f[0] - first_step / 2.0
    edges[-1] = depth_f[-1] + last_step / 2.0
    return edges


def compress_intervals(points: list[ClassifiedPoint], edges: np.ndarray) -> list[ClassifiedInterval]:
    """Объединяет соседние точки одного класса в непрерывные интервалы."""
    intervals: list[ClassifiedInterval] = []
    if not points:
        return intervals

    start_idx = 0
    current_key = points[0].class_key
    current_name = points[0].class_name
    current_lithotype = points[0].lithotype_code

    # Сжимаем соседние точки с одинаковым классом в единый интервал.
    for idx in range(1, len(points)):
        p = points[idx]
        if p.class_key == current_key and p.class_name == current_name and p.lithotype_code == current_lithotype:
            continue
        top = float(edges[start_idx])
        bottom = float(edges[idx])
        thickness = abs(bottom - top)
        intervals.append(
            ClassifiedInterval(
                start_depth=min(top, bottom),
                end_depth=max(top, bottom),
                thickness=thickness,
                lithotype_code=current_lithotype,
                class_key=current_key,
                class_name=current_name,
                sample_count=idx - start_idx,
            )
        )
        start_idx = idx
        current_key = p.class_key
        current_name = p.class_name
        current_lithotype = p.lithotype_code

    top = float(edges[start_idx])
    bottom = float(edges[len(points)])
    intervals.append(
        ClassifiedInterval(
            start_depth=min(top, bottom),
            end_depth=max(top, bottom),
            thickness=abs(bottom - top),
            lithotype_code=current_lithotype,
            class_key=current_key,
            class_name=current_name,
            sample_count=len(points) - start_idx,
        )
    )
    return intervals


def plot_well_column(
    well_name: str,
    intervals: list[ClassifiedInterval],
    output_png: Path,
    depth_min: float,
    depth_max: float,
) -> None:
    """Рисует финальную литологическую колонку для одной скважины."""
    fig, (ax_lithotype, ax_main) = plt.subplots(
        ncols=2,
        figsize=(8.5, 12),
        sharey=True,
        gridspec_kw={"width_ratios": [0.8, 3.2]},
    )

    for iv in intervals:
        y = iv.start_depth
        h = iv.end_depth - iv.start_depth
        style = STYLES.get(iv.class_key, STYLES["undefined"])
        ax_main.add_patch(
            Rectangle(
                (0.0, y),
                1.0,
                h,
                facecolor=style.color,
                edgecolor="#2d2d2d",
                linewidth=0.45,
                hatch=style.hatch,
            )
        )
        litho_color = "#d9975d" if iv.lithotype_code == 1 else "#768fa9"
        ax_lithotype.add_patch(
            Rectangle(
                (0.0, y),
                1.0,
                h,
                facecolor=litho_color,
                edgecolor="#2d2d2d",
                linewidth=0.45,
            )
        )

    for ax in (ax_lithotype, ax_main):
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_ylim(depth_max, depth_min)

    ax_lithotype.set_title("Литотип")
    ax_main.set_title("Финальная литология")
    ax_lithotype.set_ylabel("Глубина")
    ax_main.set_xlabel(well_name)
    fig.suptitle(f"Финальная литологическая колонка — {well_name}", fontsize=13)

    handles_main: list[Patch] = []
    used_keys = sorted({iv.class_key for iv in intervals})
    for key in used_keys:
        style = STYLES.get(key, STYLES["undefined"])
        handles_main.append(
            Patch(facecolor=style.color, edgecolor="#2d2d2d", hatch=style.hatch, label=style.label)
        )
    handles_litho = [
        Patch(facecolor="#d9975d", edgecolor="#2d2d2d", label="1: коллектор"),
        Patch(facecolor="#768fa9", edgecolor="#2d2d2d", label="2: покрышка"),
    ]
    legend_handles = handles_litho + handles_main
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    payload: list[tuple[str, list[ClassifiedInterval], float, float]],
    output_png: Path,
) -> None:
    """Рисует сводную колонку по всем скважинам на одном полотне."""
    if not payload:
        return

    grouped_intervals: dict[str, list[ClassifiedInterval]] = {}
    grouped_segments: dict[str, list[str]] = {}
    for well_name, intervals, _depth_min, _depth_max in payload:
        well_group = well_name.split("_", 1)[0]
        grouped_intervals.setdefault(well_group, []).extend(intervals)
        grouped_segments.setdefault(well_group, []).append(well_name)

    if not grouped_intervals:
        return

    for well_group in grouped_intervals:
        grouped_intervals[well_group].sort(key=lambda iv: (iv.start_depth, iv.end_depth))

    global_depth_min = min(iv.start_depth for grouped in grouped_intervals.values() for iv in grouped)
    global_depth_max = max(iv.end_depth for grouped in grouped_intervals.values() for iv in grouped)

    ordered_groups = sorted(grouped_intervals.keys())
    n = len(ordered_groups)
    fig, axes = plt.subplots(
        ncols=n,
        figsize=(max(12.0, 1.1 * n + 3.0), 11),
        sharey=True,
        squeeze=False,
    )
    ax_list = list(axes[0])

    for idx, well_group in enumerate(ordered_groups):
        ax = ax_list[idx]
        intervals = grouped_intervals[well_group]
        for iv in intervals:
            style = STYLES.get(iv.class_key, STYLES["undefined"])
            ax.add_patch(
                Rectangle(
                    (0.0, iv.start_depth),
                    1.0,
                    iv.end_depth - iv.start_depth,
                    facecolor=style.color,
                    edgecolor="#1f1f1f",
                    linewidth=0.25,
                    hatch=style.hatch,
                )
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_ylim(global_depth_max, global_depth_min)
        segment_count = len(grouped_segments[well_group])
        if segment_count > 1:
            ax.set_title(f"{well_group}\n({segment_count} разреза)", fontsize=8)
        else:
            ax.set_title(well_group, fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.45)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    ax_list[0].set_ylabel("Глубина")

    handles = [
        Patch(facecolor=style.color, edgecolor="#2d2d2d", hatch=style.hatch, label=style.label)
        for _, style in STYLES.items()
        if style.label != STYLES["undefined"].label
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle("Финальные литологические колонки по всем скважинам", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_points_csv(path: Path, points: list[ClassifiedPoint]) -> None:
    """Сохраняет построчную классификацию точки-в-точку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "depth",
                "lithotype_code",
                "kaolinite",
                "hydromica",
                "feldspar",
                "quartz",
                "porosity",
                "clay_share_solid",
                "q_over_qf",
                "porosity_tag",
                "class_key",
                "class_name",
            ]
        )
        for p in points:
            writer.writerow(
                [
                    f"{p.depth:.4f}",
                    p.lithotype_code,
                    f"{p.kaolinite:.6f}",
                    f"{p.hydromica:.6f}",
                    f"{p.feldspar:.6f}",
                    f"{p.quartz:.6f}",
                    f"{p.porosity:.6f}",
                    f"{p.clay_share:.6f}",
                    f"{p.q_over_qf:.6f}",
                    p.porosity_tag,
                    p.class_key,
                    p.class_name,
                ]
            )


def write_intervals_csv(path: Path, intervals: list[ClassifiedInterval]) -> None:
    """Сохраняет интервальную классификацию после сжатия соседних точек."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_depth",
                "end_depth",
                "thickness",
                "lithotype_code",
                "class_key",
                "class_name",
                "sample_count",
            ]
        )
        for iv in intervals:
            writer.writerow(
                [
                    f"{iv.start_depth:.4f}",
                    f"{iv.end_depth:.4f}",
                    f"{iv.thickness:.4f}",
                    iv.lithotype_code,
                    iv.class_key,
                    iv.class_name,
                    iv.sample_count,
                ]
            )


def parse_args() -> argparse.Namespace:
    """Описывает CLI-аргументы построения финальной литологии."""
    p = argparse.ArgumentParser(
        description="Построить финальные литологические колонки по всем файлам *_mkm_ga_plot_data.npz."
    )
    p.add_argument("--project-root", default="", help="Корень проекта (по умолчанию auto).")
    p.add_argument(
        "--input-root",
        default="outputs/mkm_result",
        help="Каталог со скважинами и *_mkm_ga_plot_data.npz.",
    )
    p.add_argument(
        "--output-root",
        default="outputs/final_lithology",
        help="Каталог для финальных CSV/PNG.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Список имен скважин для фильтрации (например: 621_1700_1780 622_1612_1696).",
    )
    return p.parse_args()


def main() -> None:
    """Точка входа: строит финальную литологию по всем GA-архивам."""
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    input_root = resolve_path(args.input_root, project_root)
    output_root = resolve_path(args.output_root, project_root)

    # Берем только архивы, сформированные после GA, по одной подпапке на скважину.
    npz_files = sorted(input_root.glob("*/*_mkm_ga_plot_data.npz"))
    if args.only:
        only_set = set(args.only)
        npz_files = [p for p in npz_files if p.parent.name in only_set]
    if not npz_files:
        raise FileNotFoundError(f"Не найдено файлов *_mkm_ga_plot_data.npz в {input_root}")

    summary_rows: list[list[object]] = []
    overview_payload: list[tuple[str, list[ClassifiedInterval], float, float]] = []

    total_points = 0
    total_intervals = 0
    for npz_path in npz_files:
        well_name = npz_path.parent.name
        with np.load(npz_path) as d:
            if "mkm_model" not in d:
                print(f"[WARN] Пропуск {npz_path}: нет ключа mkm_model")
                continue
            mkm_model = np.asarray(d["mkm_model"], dtype=float)

        if mkm_model.ndim != 2 or mkm_model.shape[1] != 7:
            print(f"[WARN] Пропуск {npz_path}: mkm_model shape={mkm_model.shape}, ожидалось (N, 7)")
            continue
        if mkm_model.shape[0] == 0:
            print(f"[WARN] Пропуск {npz_path}: пустой mkm_model")
            continue

        # Классификация "точка за точкой", затем агрегирование в интервалы для колонок.
        points = [classify_row(row) for row in mkm_model]
        edges = build_depth_edges(np.array([p.depth for p in points], dtype=float))
        intervals = compress_intervals(points, edges)

        well_out = output_root / well_name
        points_csv = well_out / f"{well_name}_final_lithology_points.csv"
        intervals_csv = well_out / f"{well_name}_final_lithology_intervals.csv"
        column_png = well_out / f"{well_name}_final_lithology_column.png"

        write_points_csv(points_csv, points)
        write_intervals_csv(intervals_csv, intervals)
        depth_min = float(np.nanmin(edges))
        depth_max = float(np.nanmax(edges))
        plot_well_column(well_name, intervals, column_png, depth_min=depth_min, depth_max=depth_max)

        class_counts: dict[str, float] = {}
        for iv in intervals:
            class_counts[iv.class_name] = class_counts.get(iv.class_name, 0.0) + iv.thickness
        dominant_class, dominant_thickness = max(class_counts.items(), key=lambda kv: kv[1])

        summary_rows.append(
            [
                well_name,
                mkm_model.shape[0],
                len(intervals),
                f"{depth_min:.2f}",
                f"{depth_max:.2f}",
                dominant_class,
                f"{dominant_thickness:.2f}",
                str(points_csv.relative_to(project_root)),
                str(intervals_csv.relative_to(project_root)),
                str(column_png.relative_to(project_root)),
            ]
        )
        overview_payload.append((well_name, intervals, depth_min, depth_max))
        total_points += mkm_model.shape[0]
        total_intervals += len(intervals)

    if not summary_rows:
        raise RuntimeError("Нет обработанных скважин. Проверьте входные данные.")

    summary_csv = output_root / "final_lithology_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "well_name",
                "n_points",
                "n_intervals",
                "depth_min",
                "depth_max",
                "dominant_class",
                "dominant_thickness",
                "points_csv",
                "intervals_csv",
                "column_png",
            ]
        )
        writer.writerows(summary_rows)

    overview_png = output_root / "final_lithology_overview.png"
    plot_overview(sorted(overview_payload, key=lambda x: x[0]), overview_png)

    print(f"Скважин обработано: {len(summary_rows)}")
    print(f"Точек классифицировано: {total_points}")
    print(f"Интервалов выделено: {total_intervals}")
    print(f"Сводка: {summary_csv}")
    print(f"Обзор: {overview_png}")


if __name__ == "__main__":
    main()
