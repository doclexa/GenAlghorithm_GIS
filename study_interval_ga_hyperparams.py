from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mkm_core import (
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    load_mkm_from_las,
    resolve_path,
    split_lithotype_intervals,
    validate_matrix_shape,
)
from mkm_ga_engine import (
    DEFAULT_GA_CXPB,
    DEFAULT_GA_INDPB,
    DEFAULT_GA_MUTPB,
    DEFAULT_GA_PATIENCE,
    DEFAULT_GA_POPULATION_SIZE,
    DEFAULT_GA_TOURNSIZE,
    GAParams,
)
from mkm_interval_optimizer import (
    IntervalGenerationQualityPoint,
    run_interval_ga,
)


@dataclass
class StudyCurve:
    parameter: str
    value: float
    points: list[IntervalGenerationQualityPoint]
    total_time_sec: float


def parse_int_list(csv_values: str) -> list[int]:
    values = [item.strip() for item in csv_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Список целых значений пуст.")
    return [int(value) for value in values]


def parse_float_list(csv_values: str) -> list[float]:
    values = [item.strip() for item in csv_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Список вещественных значений пуст.")
    return [float(value) for value in values]


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def cleanup_output_dir(output_dir: Path) -> None:
    patterns = ("ga_effect_*.png", "ga_hyperparam_study_*.csv")
    removed = 0
    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed += 1
    if removed:
        print(f"Удалено старых артефактов: {removed}")


def make_ga_params(base_params: GAParams, parameter: str, value: float) -> GAParams:
    params_dict = asdict(base_params)
    if parameter in {"population_size", "tournsize", "ngen", "patience"}:
        params_dict[parameter] = int(value)
    else:
        params_dict[parameter] = float(value)
    return GAParams(**params_dict)


def run_single_curve(
    *,
    data: np.ndarray,
    intervals,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    ga_params: GAParams,
) -> tuple[list[IntervalGenerationQualityPoint], float]:
    started = time.perf_counter()
    summary = run_interval_ga(
        data=data,
        intervals=intervals,
        a_min_coll=a_min_coll,
        a_max_coll=a_max_coll,
        a_min_glin=a_min_glin,
        a_max_glin=a_max_glin,
        w_negative=w_negative,
        w_glin=w_glin,
        w_coll=w_coll,
        ga_params=ga_params,
        verbose=False,
    )
    elapsed_sec = time.perf_counter() - started
    if summary.quality_curve:
        points = summary.quality_curve
    else:
        points = [
            IntervalGenerationQualityPoint(
                generation=0,
                quality_score=summary.quality_score,
                negative_share=summary.negative_share,
                glin_bad_share=summary.glin_bad_share,
                coll_bad_share=summary.coll_bad_share,
                tested_matrices=summary.total_evals,
            )
        ]
    return points, elapsed_sec


def save_curves_csv(path: Path, all_curves: list[StudyCurve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "parameter",
                "value",
                "generation",
                "quality_score",
                "negative_share",
                "glin_bad_share",
                "coll_bad_share",
                "tested_matrices",
            ]
        )
        for curve in all_curves:
            for point in curve.points:
                writer.writerow(
                    [
                        curve.parameter,
                        curve.value,
                        point.generation,
                        f"{point.quality_score:.10f}",
                        f"{point.negative_share:.10f}",
                        f"{point.glin_bad_share:.10f}",
                        f"{point.coll_bad_share:.10f}",
                        point.tested_matrices,
                    ]
                )


def save_summary_csv(path: Path, all_curves: list[StudyCurve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "parameter",
                "value",
                "final_generation",
                "final_quality_score",
                "total_tested_matrices",
                "total_time_sec",
            ]
        )
        for curve in all_curves:
            final_point = curve.points[-1]
            writer.writerow(
                [
                    curve.parameter,
                    curve.value,
                    final_point.generation,
                    f"{final_point.quality_score:.10f}",
                    final_point.tested_matrices,
                    f"{curve.total_time_sec:.6f}",
                ]
            )


def load_curves_csv(path: Path) -> dict[tuple[str, float], list[IntervalGenerationQualityPoint]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден CSV с кривыми: {path}. Сначала выполните полный запуск без --plot-only."
        )

    grouped: dict[tuple[str, float], list[IntervalGenerationQualityPoint]] = {}
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {
            "parameter",
            "value",
            "generation",
            "quality_score",
            "negative_share",
            "glin_bad_share",
            "coll_bad_share",
            "tested_matrices",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В CSV кривых отсутствуют колонки: {sorted(missing)}")

        for row in reader:
            key = (row["parameter"], float(row["value"]))
            grouped.setdefault(key, []).append(
                IntervalGenerationQualityPoint(
                    generation=int(row["generation"]),
                    quality_score=float(row["quality_score"]),
                    negative_share=float(row["negative_share"]),
                    glin_bad_share=float(row["glin_bad_share"]),
                    coll_bad_share=float(row["coll_bad_share"]),
                    tested_matrices=int(row["tested_matrices"]),
                )
            )

    for points in grouped.values():
        points.sort(key=lambda point: point.generation)
    return grouped


def load_summary_csv(path: Path) -> dict[tuple[str, float], float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден CSV summary: {path}. Сначала выполните полный запуск без --plot-only."
        )

    summary_time: dict[tuple[str, float], float] = {}
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {
            "parameter",
            "value",
            "final_generation",
            "final_quality_score",
            "total_tested_matrices",
            "total_time_sec",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В CSV summary отсутствуют колонки: {sorted(missing)}")

        for row in reader:
            key = (row["parameter"], float(row["value"]))
            summary_time[key] = float(row["total_time_sec"])
    return summary_time


def group_curves_by_parameter(all_curves: list[StudyCurve]) -> dict[str, list[StudyCurve]]:
    grouped: dict[str, list[StudyCurve]] = {}
    for curve in all_curves:
        grouped.setdefault(curve.parameter, []).append(curve)
    return grouped


def save_parameter_q_generation_plot(
    *,
    parameter: str,
    curves: list[StudyCurve],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for curve in curves:
        generations = [point.generation for point in curve.points]
        quality_scores = [point.quality_score for point in curve.points]
        ax.plot(
            generations,
            quality_scores,
            label=f"{parameter}={format_value(curve.value)}",
        )

    ax.set_xlabel("Поколение")
    ax.set_ylabel("Q")
    ax.set_title(f"Влияние {parameter} на Q по поколениям")
    ax.grid(True)
    ax.legend(loc="best")

    footer_parts = []
    for curve in curves:
        footer_parts.append(
            f"{parameter}={format_value(curve.value)}: {curve.points[-1].tested_matrices}"
        )
    footer = "Перебрано матриц (итог): " + " | ".join(footer_parts)
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10, wrap=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_all_plots(all_curves: list[StudyCurve], output_dir: Path) -> None:
    grouped = group_curves_by_parameter(all_curves)
    for parameter, curves in grouped.items():
        curves_sorted = sorted(curves, key=lambda item: item.value)
        save_parameter_q_generation_plot(
            parameter=parameter,
            curves=curves_sorted,
            output_path=output_dir / f"ga_effect_{parameter}_Q_vs_generation.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Строгое исследование влияния гиперпараметров интервального GA "
            "на качество Q по поколениям для выбранной скважины."
        )
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Корень проекта (по умолчанию каталог, где лежит mkm_core.py).",
    )
    parser.add_argument(
        "--las",
        default=DEFAULT_LAS_RELPATH,
        help="Путь к .las относительно корня проекта или абсолютный.",
    )
    parser.add_argument("--depth", default="DEPT", help="Мнемоника глубины в LAS.")
    parser.add_argument("--litho", default="LITO", help="Мнемоника литологии в LAS.")
    parser.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства для МКМ (опционально; иначе POTA THOR RHOB WNKT).",
    )
    parser.add_argument("--config-dir", default="config", help="Каталог с границами матриц.")
    parser.add_argument("--a-min-coll", default="a_min_coll.in")
    parser.add_argument("--a-max-coll", default="a_max_coll.in")
    parser.add_argument("--a-min-glin", default="a_min_glin.in")
    parser.add_argument("--a-max-glin", default="a_max_glin.in")

    parser.add_argument("--w-negative", type=float, default=0.8)
    parser.add_argument("--w-glin", type=float, default=0.1)
    parser.add_argument("--w-coll", type=float, default=0.1)

    parser.add_argument("--population-size", type=int, default=DEFAULT_GA_POPULATION_SIZE)
    parser.add_argument("--ngen", type=int, default=110)
    parser.add_argument("--cxpb", type=float, default=DEFAULT_GA_CXPB)
    parser.add_argument("--mutpb", type=float, default=DEFAULT_GA_MUTPB)
    parser.add_argument("--indpb", type=float, default=DEFAULT_GA_INDPB)
    parser.add_argument("--tournsize", type=int, default=DEFAULT_GA_TOURNSIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_GA_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--population-values", default="140,220,320,400,420")
    parser.add_argument("--cxpb-values", default="0.4,0.55,0.7,0.8,0.85")
    parser.add_argument("--mutpb-values", default="0.1,0.25,0.4,0.55")
    parser.add_argument("--indpb-values", default="0.04,0.10,0.18,0.28,0.5")
    parser.add_argument("--tournsize-values", default="2,3,5,7")
    parser.add_argument("--ngen-values", default="60,110,170,240")
    parser.add_argument("--patience-values", default="0,10,25,50")

    parser.add_argument(
        "--output-dir",
        default="outputs/ga_hyperparam_study_skv621_run",
        help="Каталог для CSV и графиков.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Не запускать оптимизацию, а только перерисовать графики из CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    output_dir = resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_csv_path = output_dir / "ga_hyperparam_study_q_curves.csv"
    summary_csv_path = output_dir / "ga_hyperparam_study_q_summary.csv"

    if args.plot_only:
        grouped_points = load_curves_csv(curves_csv_path)
        summary_time = load_summary_csv(summary_csv_path)
        all_curves: list[StudyCurve] = []
        for key, points in grouped_points.items():
            if key not in summary_time:
                raise ValueError(
                    f"Для {key} нет строки в summary CSV. "
                    "Перезапустите полный sweep без --plot-only."
                )
            parameter, value = key
            all_curves.append(
                StudyCurve(
                    parameter=parameter,
                    value=value,
                    points=points,
                    total_time_sec=summary_time[key],
                )
            )
        render_all_plots(all_curves, output_dir)
        print(f"Графики перерисованы из CSV в каталоге: {output_dir}")
        return

    cleanup_output_dir(output_dir)

    sweep_values = {
        "population_size": [float(v) for v in parse_int_list(args.population_values)],
        "cxpb": parse_float_list(args.cxpb_values),
        "mutpb": parse_float_list(args.mutpb_values),
        "indpb": parse_float_list(args.indpb_values),
        "tournsize": [float(v) for v in parse_int_list(args.tournsize_values)],
        "ngen": [float(v) for v in parse_int_list(args.ngen_values)],
        "patience": [float(v) for v in parse_int_list(args.patience_values)],
    }

    las_path = resolve_path(args.las, project_root)
    config_dir = resolve_path(args.config_dir, project_root)
    props = tuple(args.props) if args.props is not None else None
    data, _is_coll, _is_glin, _coll_prop, _glin_prop, _litho_raw = load_mkm_from_las(
        las_path,
        depth_mnem=args.depth,
        litho_mnem=args.litho,
        prop_mnems=props,
        verbose=True,
    )
    intervals = split_lithotype_intervals(data)

    a_min_coll = np.loadtxt(config_dir / args.a_min_coll)
    a_max_coll = np.loadtxt(config_dir / args.a_max_coll)
    a_min_glin = np.loadtxt(config_dir / args.a_min_glin)
    a_max_glin = np.loadtxt(config_dir / args.a_max_glin)

    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    base_params = GAParams(
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

    print("Старт строгого исследования гиперпараметров интервального GA.")
    print(
        f"База: pop={base_params.population_size}, ngen={base_params.ngen}, "
        f"cxpb={base_params.cxpb}, mutpb={base_params.mutpb}, indpb={base_params.indpb}, "
        f"tournsize={base_params.tournsize}, patience={base_params.patience}"
    )
    print(
        f"Q = {args.w_negative}*negative + "
        f"{args.w_glin}*glin_bad + {args.w_coll}*coll_bad"
    )

    all_curves: list[StudyCurve] = []
    for parameter, values in sweep_values.items():
        print(f"\nПараметр: {parameter} | значения: {[format_value(v) for v in values]}")
        for idx, value in enumerate(values, start=1):
            ga_params = make_ga_params(base_params, parameter, value)
            points, elapsed_sec = run_single_curve(
                data=data,
                intervals=intervals,
                a_min_coll=a_min_coll,
                a_max_coll=a_max_coll,
                a_min_glin=a_min_glin,
                a_max_glin=a_max_glin,
                w_negative=args.w_negative,
                w_glin=args.w_glin,
                w_coll=args.w_coll,
                ga_params=ga_params,
            )
            curve = StudyCurve(
                parameter=parameter,
                value=float(value),
                points=points,
                total_time_sec=elapsed_sec,
            )
            all_curves.append(curve)
            final_point = curve.points[-1]
            print(
                f"  {idx}/{len(values)} value={format_value(value)} | "
                f"gen={final_point.generation} | Q={final_point.quality_score:.8f} | "
                f"tested={final_point.tested_matrices} | time={elapsed_sec:.2f}s"
            )

    save_curves_csv(curves_csv_path, all_curves)
    save_summary_csv(summary_csv_path, all_curves)
    render_all_plots(all_curves, output_dir)

    print("\nИсследование завершено.")
    print(f"CSV кривых: {curves_csv_path}")
    print(f"CSV summary: {summary_csv_path}")
    print(f"Графики: {output_dir}")


if __name__ == "__main__":
    main()
