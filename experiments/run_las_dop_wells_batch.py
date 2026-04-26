#!/usr/bin/env python3
"""Пакетный прогон интервального BF (splitting) и GA по каталогу .las (например data/las_dop_wells)."""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkm_core import (  # noqa: E402
    load_mkm_from_las,
    resolve_path,
    save_mkm_plot,
    scale_mkm_model_for_metrics,
    split_lithotype_intervals,
    validate_k_shape,
    validate_matrix_shape,
)
from mkm_ga_engine import GAParams  # noqa: E402
from mkm_interval_optimizer import (  # noqa: E402
    IntervalOptimizationSummary,
    apply_k_splitting,
    run_interval_bruteforce,
    run_interval_ga,
    save_interval_matrices_npz,
    write_interval_results_csv,
)


def _discover_las_files(las_dir: Path) -> list[Path]:
    paths = sorted(las_dir.glob("*.las"))
    return [p for p in paths if p.is_file()]


def _format_bf_section(*, bf_splitting: int, bf: IntervalOptimizationSummary) -> str:
    lines = [
        "[bruteforce]",
        f"  splitting: {bf_splitting}",
        f"  quality_score Q: {bf.quality_score:.10f}",
        f"  negative_share: {bf.negative_share:.10f}",
        f"  glin_bad_share: {bf.glin_bad_share:.10f}",
        f"  coll_bad_share: {bf.coll_bad_share:.10f}",
        f"  time_sec: {bf.total_time_sec:.6f}",
        f"  total_evals: {bf.total_evals}",
        f"  total_invalid_count: {bf.total_invalid_count}",
        f"  intervals: {len(bf.interval_results)}",
    ]
    return "\n".join(lines)


def _format_ga_section(*, ga_params: GAParams, ga: IntervalOptimizationSummary) -> str:
    lines = [
        "[ga]",
        f"  population_size: {ga_params.population_size}",
        f"  ngen_cap: {ga_params.ngen}",
        f"  cxpb: {ga_params.cxpb}",
        f"  mutpb: {ga_params.mutpb}",
        f"  indpb: {ga_params.indpb}",
        f"  tournsize: {ga_params.tournsize}",
        f"  patience: {ga_params.patience}",
        f"  min_delta: {ga_params.min_delta}",
        f"  n_jobs: {ga_params.n_jobs}",
        f"  seed: {ga_params.seed}",
        f"  quality_score Q: {ga.quality_score:.10f}",
        f"  negative_share: {ga.negative_share:.10f}",
        f"  glin_bad_share: {ga.glin_bad_share:.10f}",
        f"  coll_bad_share: {ga.coll_bad_share:.10f}",
        f"  time_sec: {ga.total_time_sec:.6f}",
        f"  total_evals: {ga.total_evals}",
        f"  total_generations: {ga.total_generations}",
        f"  intervals: {len(ga.interval_results)}",
    ]
    return "\n".join(lines)


def _format_metrics(
    stem: str,
    las_rel: str,
    *,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    bf_splitting: int,
    bf: IntervalOptimizationSummary | None,
    ga_params: GAParams,
    ga: IntervalOptimizationSummary,
) -> str:
    header = [
        f"well_stem: {stem}",
        f"las: {las_rel}",
        f"weights: w_negative={w_negative} w_glin={w_glin} w_coll={w_coll}",
        "",
    ]
    if bf is not None:
        body_bf = _format_bf_section(bf_splitting=bf_splitting, bf=bf)
    else:
        body_bf = "[bruteforce]\n  note: не пересчитывался (--ga-only)"
    body_ga = _format_ga_section(ga_params=ga_params, ga=ga)
    return "\n".join(header) + body_bf + "\n\n" + body_ga + "\n"


def _merge_metrics_keep_bf_head(existing_text: str, new_tail_from_ga: str) -> str:
    """Сохраняет текст до секции [ga] и подставляет новую секцию [ga] + метрики."""
    m = re.search(r"^\[ga\]\s*$", existing_text, re.MULTILINE)
    if m:
        head = existing_text[: m.start()].rstrip()
        return head + "\n\n" + new_tail_from_ga.strip() + "\n"
    return existing_text.rstrip() + "\n\n" + new_tail_from_ga.strip() + "\n"


def _process_one_well(
    *,
    las_path: Path,
    project_root: Path,
    config_dir: Path,
    well_dir: Path,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    splitting: int,
    ga_params: GAParams,
    depth: str,
    litho: str,
    prop_mnems: tuple[str, str, str, str] | None,
    a_names: dict[str, str],
    opt_verbose: bool,
    ga_only: bool,
) -> None:
    well_dir.mkdir(parents=True, exist_ok=True)
    props = prop_mnems

    data, _is_c, _is_g, _cp, _gp, litho_raw = load_mkm_from_las(
        las_path,
        depth_mnem=depth,
        litho_mnem=litho,
        prop_mnems=props,
        verbose=True,
    )
    intervals = split_lithotype_intervals(data)

    a_min_coll = np.loadtxt(config_dir / a_names["a_min_coll"])
    a_max_coll = np.loadtxt(config_dir / a_names["a_max_coll"])
    a_min_glin = np.loadtxt(config_dir / a_names["a_min_glin"])
    a_max_glin = np.loadtxt(config_dir / a_names["a_max_glin"])

    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    if not ga_only:
        a_k_coll = np.loadtxt(config_dir / a_names["a_k_coll"])
        a_k_glin = np.loadtxt(config_dir / a_names["a_k_glin"])
        validate_k_shape(a_k_coll, "A_k_coll")
        validate_k_shape(a_k_glin, "A_k_glin")
        a_k_coll_eff = apply_k_splitting(a_k_coll, splitting)
        a_k_glin_eff = apply_k_splitting(a_k_glin, splitting)

    stem = las_path.stem
    try:
        rel_las = str(las_path.relative_to(project_root))
    except ValueError:
        rel_las = str(las_path)

    bf_summary: IntervalOptimizationSummary | None = None
    if not ga_only:
        print(f"\n=== BF: {stem} (splitting={splitting}) ===", flush=True)
        bf_summary = run_interval_bruteforce(
            data=data,
            intervals=intervals,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_k_coll=a_k_coll_eff,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            a_k_glin=a_k_glin_eff,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            max_iterations=None,
            verbose=opt_verbose,
        )
        plot_bf = well_dir / f"{stem}_mkm_bf.png"
        npz_bf = well_dir / f"{stem}_intervals_bf.npz"
        csv_bf = well_dir / f"{stem}_bf_interval_metrics.csv"
        save_interval_matrices_npz(bf_summary, npz_bf)
        write_interval_results_csv(bf_summary.interval_results, csv_bf)
        mkm_bf = scale_mkm_model_for_metrics(bf_summary.mkm_model)
        save_mkm_plot(
            mkm_bf,
            plot_bf,
            litho_raw=litho_raw,
            litho_mnem=litho,
            intervals=intervals,
        )
    else:
        print(f"\n=== BF: {stem} — пропуск (--ga-only) ===", flush=True)

    print(f"\n=== GA: {stem} ===", flush=True)
    ga_summary = run_interval_ga(
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
        verbose=opt_verbose,
    )
    plot_ga = well_dir / f"{stem}_mkm_ga.png"
    npz_ga = well_dir / f"{stem}_intervals_ga.npz"
    csv_ga = well_dir / f"{stem}_ga_interval_metrics.csv"
    save_interval_matrices_npz(ga_summary, npz_ga)
    write_interval_results_csv(ga_summary.interval_results, csv_ga)
    mkm_ga = scale_mkm_model_for_metrics(ga_summary.mkm_model)
    save_mkm_plot(
        mkm_ga,
        plot_ga,
        litho_raw=litho_raw,
        litho_mnem=litho,
        intervals=intervals,
    )

    metrics_path = well_dir / "metrics.txt"
    ga_section = _format_ga_section(ga_params=ga_params, ga=ga_summary)
    if ga_only and metrics_path.is_file():
        existing = metrics_path.read_text(encoding="utf-8")
        text = _merge_metrics_keep_bf_head(existing, ga_section)
    else:
        text = _format_metrics(
            stem,
            rel_las,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            bf_splitting=splitting,
            bf=bf_summary,
            ga_params=ga_params,
            ga=ga_summary,
        )
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Готово: {well_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BF (splitting) + GA по всем .las в каталоге; вывод в outputs/new_wells/<stem>/."
    )
    p.add_argument(
        "--ga-only",
        action="store_true",
        help="Только GA: не пересчитывать BF (файлы *_mkm_bf.* не трогаются). "
        "В metrics.txt секция [bruteforce] сохраняется из существующего файла, если он есть.",
    )
    p.add_argument("--project-root", default="", help="Корень проекта (по умолчанию — родитель experiments/).")
    p.add_argument("--las-dir", default="data/las_dop_wells", help="Каталог с .las.")
    p.add_argument("--out-root", default="outputs/new_wells", help="Корень вывода (подпапка на скважину).")
    p.add_argument("--config-dir", default="config", help="Каталог с a_*.in.")
    p.add_argument("--depth", default="DEPT", help="Мнемоника глубины.")
    p.add_argument("--litho", default="LITO", help="Мнемоника литологии.")
    p.add_argument(
        "--props",
        nargs=4,
        metavar="MNEM",
        default=None,
        help="Четыре кривые-свойства; иначе авто, как в mkm_run_bruteforce.",
    )
    p.add_argument("--a-min-coll", default="a_min_coll.in")
    p.add_argument("--a-max-coll", default="a_max_coll.in")
    p.add_argument("--a-k-coll", default="a_k_coll.in")
    p.add_argument("--a-min-glin", default="a_min_glin.in")
    p.add_argument("--a-max-glin", default="a_max_glin.in")
    p.add_argument("--a-k-glin", default="a_k_glin.in")
    p.add_argument("--w-negative", type=float, default=0.8)
    p.add_argument("--w-glin", type=float, default=0.1)
    p.add_argument("--w-coll", type=float, default=0.1)
    p.add_argument(
        "--splitting",
        type=int,
        choices=(2, 3, 4, 5),
        default=4,
        help="Узлы linspace для BF (a_k, ячейки != 1).",
    )
    p.add_argument("--population-size", type=int, default=220)
    p.add_argument("--ngen", type=int, default=10000, help="Верхняя граница поколений GA (факт — early stop).")
    p.add_argument("--cxpb", type=float, default=0.6)
    p.add_argument("--mutpb", type=float, default=0.25)
    p.add_argument("--indpb", type=float, default=0.1)
    p.add_argument("--tournsize", type=int, default=3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Список стемов файлов (без .las); если пусто — все .las в каталоге.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="При ошибке на одной скважине печатать traceback и продолжать.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод перебора/GA по интервалам (иначе к оптимизаторам verbose=False).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else PROJECT_ROOT
    )
    las_dir = resolve_path(args.las_dir, project_root)
    out_root = resolve_path(args.out_root, project_root)
    config_dir = resolve_path(args.config_dir, project_root)

    a_names = {
        "a_min_coll": args.a_min_coll,
        "a_max_coll": args.a_max_coll,
        "a_k_coll": args.a_k_coll,
        "a_min_glin": args.a_min_glin,
        "a_max_glin": args.a_max_glin,
        "a_k_glin": args.a_k_glin,
    }

    ga_params = GAParams(
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
    opt_verbose = bool(args.verbose)
    prop_tuple = tuple(args.props) if args.props is not None else None

    all_las = _discover_las_files(las_dir)
    if not all_las:
        print(f"Нет .las в {las_dir}", file=sys.stderr)
        sys.exit(1)

    only = set(args.only) if args.only else None
    if only:
        by_stem = {p.stem: p for p in all_las}
        for name in only:
            if name not in by_stem:
                print(
                    f"Предупреждение: --only {name} — файла {name}.las нет в {las_dir}",
                    file=sys.stderr,
                )
        to_run = [by_stem[s] for s in sorted(only) if s in by_stem]
    else:
        to_run = all_las

    if not to_run:
        print("Список скважин пуст после фильтра --only.", file=sys.stderr)
        sys.exit(1)

    mode = "GA-only" if args.ga_only else "BF+GA"
    print(f"К обработке: {len(to_run)} скваж. режим={mode} out={out_root}", flush=True)

    for las_path in to_run:
        stem = las_path.stem
        well_dir = out_root / stem
        try:
            _process_one_well(
                las_path=las_path,
                project_root=project_root,
                config_dir=config_dir,
                well_dir=well_dir,
                w_negative=args.w_negative,
                w_glin=args.w_glin,
                w_coll=args.w_coll,
                splitting=args.splitting,
                ga_params=ga_params,
                depth=args.depth,
                litho=args.litho,
                prop_mnems=prop_tuple,
                a_names=a_names,
                opt_verbose=opt_verbose,
                ga_only=bool(args.ga_only),
            )
        except Exception:  # noqa: BLE001
            print(f"\nОшибка: {las_path}\n", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                sys.exit(1)


if __name__ == "__main__":
    main()
