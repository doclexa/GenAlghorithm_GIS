#!/usr/bin/env python3
"""Построить BF-график МКМ для варианта с Q, близким к заданному."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mkm_core import (  # noqa: E402
    DEFAULT_LAS_RELPATH,
    PROJECT_ROOT,
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    split_lithotype_intervals,
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_interval_optimizer import run_interval_bruteforce  # noqa: E402


@dataclass(frozen=True)
class CandidateResult:
    subdivision_count: int
    max_iterations: int | None
    quality_score: float
    negative_share: float
    glin_bad_share: float
    coll_bad_share: float
    total_evals: int
    total_time_sec: float
    summary: object


def build_subdivision_k_matrix(base_k: np.ndarray, subdivisions: int) -> np.ndarray:
    if subdivisions < 1:
        raise ValueError("Количество разбиений должно быть >= 1.")
    base = np.asarray(base_k, dtype=int)
    active_mask = base > 1
    result = np.ones_like(base, dtype=int)
    result[active_mask] = int(subdivisions)
    return result


def normalize_target_q(raw_q: float) -> float:
    if raw_q <= 0:
        raise ValueError("Q должен быть положительным.")
    # Удобно принимать и 0.09, и 9 (проценты).
    return raw_q / 100.0 if raw_q > 1 else raw_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Подобрать BF-вариант с качеством Q, близким к заданному, "
            "и построить график МКМ по глубине."
        )
    )
    parser.add_argument("--project-root", default="", help="Корень проекта.")
    parser.add_argument("--las", default=DEFAULT_LAS_RELPATH, help="Путь к .las.")
    parser.add_argument("--depth", default="DEPT", help="Мнемоника глубины.")
    parser.add_argument("--litho", default="LITO", help="Мнемоника литологии.")
    parser.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства; по умолчанию используется автоподбор.",
    )
    parser.add_argument("--config-dir", default="config", help="Каталог с a_*.in.")
    parser.add_argument("--a-min-coll", default="a_min_coll.in")
    parser.add_argument("--a-max-coll", default="a_max_coll.in")
    parser.add_argument("--a-k-coll", default="a_k_coll.in")
    parser.add_argument("--a-min-glin", default="a_min_glin.in")
    parser.add_argument("--a-max-glin", default="a_max_glin.in")
    parser.add_argument("--a-k-glin", default="a_k_glin.in")
    parser.add_argument(
        "--target-q",
        type=float,
        required=True,
        help="Целевой Q. Можно задавать как 0.09 или как 9 (проценты).",
    )
    parser.add_argument(
        "--subdivisions",
        default="5",
        help="Список N через запятую; для каждого N строится bf_N.",
    )
    parser.add_argument(
        "--max-iterations-grid",
        default="1,8,32,128,512,2048,4096,8192,16384,0",
        help=(
            "Список лимитов перебора на интервал через запятую. "
            "0 означает полный перебор для соответствующего bf_N."
        ),
    )
    parser.add_argument("--w-negative", type=float, default=0.8)
    parser.add_argument("--w-glin", type=float, default=0.1)
    parser.add_argument("--w-coll", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default="outputs/plots/bad_bf",
        help="Папка для результирующего графика.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Полный путь к PNG. Если пусто, имя будет собрано автоматически.",
    )
    return parser.parse_args()


def choose_best_candidate(
    candidates: list[CandidateResult],
    target_q: float,
) -> CandidateResult:
    if not candidates:
        raise RuntimeError("Не найдено ни одного кандидата BF.")
    return min(
        candidates,
        key=lambda item: (abs(item.quality_score - target_q), item.quality_score),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    las_path = resolve_path(args.las, project_root)
    config_dir = resolve_path(args.config_dir, project_root)
    target_q = normalize_target_q(args.target_q)
    subdivisions = [int(item.strip()) for item in args.subdivisions.split(",") if item.strip()]
    max_iterations_grid = [
        int(item.strip()) for item in args.max_iterations_grid.split(",") if item.strip()
    ]
    if not subdivisions:
        raise ValueError("Нужно указать хотя бы одно значение в --subdivisions.")
    if not max_iterations_grid:
        raise ValueError("Нужно указать хотя бы одно значение в --max-iterations-grid.")

    props = tuple(args.props) if args.props is not None else None
    data, _is_coll, _is_glin, _coll_prop, _glin_prop, litho_raw = load_mkm_from_las(
        las_path,
        depth_mnem=args.depth,
        litho_mnem=args.litho,
        prop_mnems=props,
        verbose=True,
    )
    intervals = split_lithotype_intervals(data)

    a_min_coll = np.loadtxt(config_dir / args.a_min_coll)
    a_max_coll = np.loadtxt(config_dir / args.a_max_coll)
    a_k_coll = np.loadtxt(config_dir / args.a_k_coll)
    a_min_glin = np.loadtxt(config_dir / args.a_min_glin)
    a_max_glin = np.loadtxt(config_dir / args.a_max_glin)
    a_k_glin = np.loadtxt(config_dir / args.a_k_glin)

    for arr, name in (
        (a_min_coll, "A_min_coll"),
        (a_max_coll, "A_max_coll"),
        (a_min_glin, "A_min_glin"),
        (a_max_glin, "A_max_glin"),
    ):
        validate_matrix_shape(arr, name)
    validate_k_shape(a_k_coll, "A_k_coll")
    validate_k_shape(a_k_glin, "A_k_glin")

    print(f"Ищу BF-вариант, ближайший к target Q={target_q:.6f}")
    print(f"Кандидаты bf_N: {subdivisions}")
    print(f"Кандидаты max_iterations: {max_iterations_grid}")

    candidates: list[CandidateResult] = []
    for subdivision_count in subdivisions:
        print(f"\n=== bf_{subdivision_count} ===")
        coll_k = build_subdivision_k_matrix(a_k_coll, subdivision_count)
        glin_k = build_subdivision_k_matrix(a_k_glin, subdivision_count)
        for max_iterations_raw in max_iterations_grid:
            max_iterations = None if max_iterations_raw == 0 else max_iterations_raw
            label_budget = "full" if max_iterations is None else str(max_iterations)
            summary = run_interval_bruteforce(
                data=data,
                intervals=intervals,
                a_min_coll=a_min_coll,
                a_max_coll=a_max_coll,
                a_k_coll=coll_k,
                a_min_glin=a_min_glin,
                a_max_glin=a_max_glin,
                a_k_glin=glin_k,
                w_negative=args.w_negative,
                w_glin=args.w_glin,
                w_coll=args.w_coll,
                max_iterations=max_iterations,
                verbose=False,
            )
            candidate = CandidateResult(
                subdivision_count=subdivision_count,
                max_iterations=max_iterations,
                quality_score=float(summary.quality_score),
                negative_share=float(summary.negative_share),
                glin_bad_share=float(summary.glin_bad_share),
                coll_bad_share=float(summary.coll_bad_share),
                total_evals=int(summary.total_evals),
                total_time_sec=float(summary.total_time_sec),
                summary=summary,
            )
            candidates.append(candidate)
            print(
                f"max_iterations={label_budget}: "
                f"Q={candidate.quality_score:.6f} "
                f"(delta={abs(candidate.quality_score - target_q):.6f}), "
                f"evals={candidate.total_evals}, time={candidate.total_time_sec:.2f}s"
            )

    selected = choose_best_candidate(candidates, target_q)
    selected_summary = selected.summary

    output_dir = resolve_path(args.output_dir, project_root)
    if args.output:
        output_png_path = resolve_path(args.output, project_root)
    else:
        target_suffix = f"{target_q:.3f}".replace(".", "_")
        actual_suffix = f"{selected.quality_score:.6f}".replace(".", "_")
        iter_suffix = "full" if selected.max_iterations is None else str(selected.max_iterations)
        output_png_path = output_dir / (
            f"{las_path.stem}_mkm_bad_bf_target_{target_suffix}"
            f"_actual_{actual_suffix}_bf_{selected.subdivision_count}"
            f"_iter_{iter_suffix}.png"
        )

    save_mkm_plot(
        selected_summary.mkm_model,
        output_png_path,
        litho_raw=litho_raw,
        litho_mnem=args.litho,
        intervals=intervals,
    )

    print("\nВыбран кандидат:")
    print(f"  bf_{selected.subdivision_count}")
    print(
        "  max_iterations: "
        f"{selected.max_iterations if selected.max_iterations is not None else 'full'}"
    )
    print(f"  target Q: {target_q:.6f}")
    print(f"  actual Q: {selected.quality_score:.6f}")
    print(f"  saved: {output_png_path}")


if __name__ == "__main__":
    main()
