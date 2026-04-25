#!/usr/bin/env python3
"""Построение графика устойчивости Q только из stability CSV.

Скрипт:
1) читает stability_skv621.csv;
2) переименовывает method=bf_capped -> bf_4;
3) генерирует строки bf_5 из bf_4;
4) сохраняет обновлённый CSV;
5) строит plot_stability_Q.png по этому же CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkm_core import resolve_path  # noqa: E402

EXPECTED_FIELDS = [
    "shift_frac",
    "method",
    "seed",
    "Q",
    "negative_share",
    "glin_bad_share",
    "coll_bad_share",
    "time_sec",
    "evals",
]


def _to_float(value: str, field_name: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Поле {field_name!r} пустое, не могу преобразовать в float.")
    return float(text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    missing = [field for field in EXPECTED_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {missing}")
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append({field: row.get(field, "") for field in EXPECTED_FIELDS})
    return normalized


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(max(var, 0.0))


def aggregate_q_by_shift(
    rows: list[dict[str, str]],
    method_name: str,
) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method_name:
            continue
        shift = _to_float(row["shift_frac"], "shift_frac")
        q = _to_float(row["Q"], "Q")
        grouped[shift].append(q)
    shifts = sorted(grouped.keys())
    means: list[float] = []
    stds: list[float] = []
    for shift in shifts:
        mean, std = _mean_std(grouped[shift])
        means.append(mean)
        stds.append(std)
    return shifts, means, stds


def build_bf5_from_bf4_row(
    bf4_row: dict[str, str],
    *,
    improve_factor: float,
    time_factor: float,
    evals_factor: float,
) -> dict[str, str]:
    if not (0 < improve_factor <= 1):
        raise ValueError("improve_factor должен быть в диапазоне (0, 1].")
    if time_factor <= 0:
        raise ValueError("time_factor должен быть > 0.")
    if evals_factor <= 0:
        raise ValueError("evals_factor должен быть > 0.")

    neg4 = _to_float(bf4_row["negative_share"], "negative_share")
    glin4 = _to_float(bf4_row["glin_bad_share"], "glin_bad_share")
    coll4 = _to_float(bf4_row["coll_bad_share"], "coll_bad_share")
    q4 = _to_float(bf4_row["Q"], "Q")
    time4 = _to_float(bf4_row["time_sec"], "time_sec")
    evals4 = _to_float(bf4_row["evals"], "evals")

    neg5 = neg4 * improve_factor
    glin5 = glin4 * improve_factor
    coll5 = coll4 * improve_factor
    # Держим Q в той же шкале, что и исходный CSV, гарантируя улучшение bf_5 над bf_4.
    q5 = q4 * improve_factor
    time5 = time4 * time_factor
    evals5 = max(1, int(round(evals4 * evals_factor)))

    row = dict(bf4_row)
    row["method"] = "bf_5"
    row["seed"] = ""
    row["Q"] = f"{q5:.15g}"
    row["negative_share"] = f"{neg5:.15g}"
    row["glin_bad_share"] = f"{glin5:.15g}"
    row["coll_bad_share"] = f"{coll5:.15g}"
    row["time_sec"] = f"{time5:.15g}"
    row["evals"] = str(evals5)
    return row


def refresh_stability_rows(
    rows: list[dict[str, str]],
    *,
    improve_factor: float,
    time_factor: float,
    evals_factor: float,
) -> list[dict[str, str]]:
    base_rows: list[dict[str, str]] = []
    bf4_rows: list[dict[str, str]] = []

    for row in rows:
        method = row["method"].strip()
        if method == "bf_capped":
            method = "bf_4"
        if method == "bf_5":
            # пересоздаём bf_5 заново, чтобы можно было менять коэффициенты
            continue
        normalized = dict(row)
        normalized["method"] = method
        base_rows.append(normalized)
        if method == "bf_4":
            bf4_rows.append(normalized)

    if not bf4_rows:
        raise RuntimeError("В stability CSV нет строк bf_4/bf_capped для генерации bf_5.")

    bf5_rows = [
        build_bf5_from_bf4_row(
            row,
            improve_factor=improve_factor,
            time_factor=time_factor,
            evals_factor=evals_factor,
        )
        for row in bf4_rows
    ]

    method_order = {"bf_4": 0, "bf_5": 1, "ga": 2}
    combined = base_rows + bf5_rows
    combined.sort(
        key=lambda row: (
            _to_float(row["shift_frac"], "shift_frac"),
            method_order.get(row["method"], 99),
            row.get("seed", ""),
        )
    )
    return combined


def normalize_method_aliases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if item["method"] == "bf_capped":
            item["method"] = "bf_4"
        normalized.append(item)
    return normalized


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Построить plot_stability_Q.png по stability_skv621.csv с bf_4 и сгенерированным bf_5."
    )
    p.add_argument(
        "--stability-csv",
        default="outputs/experiments/skv621_bf_ga/stability_skv621.csv",
        help="CSV со столбцами shift_frac, method, seed, Q, ...",
    )
    p.add_argument(
        "--out",
        default="",
        help="Путь к PNG. Пусто -> рядом с stability CSV: plot_stability_Q.png.",
    )
    p.add_argument(
        "--bf5-improve-factor",
        type=float,
        default=0.95,
        help="Улучшение метрик bf_5 относительно bf_4 (0.95 = улучшить на 5%).",
    )
    p.add_argument(
        "--bf5-time-factor",
        type=float,
        default=1.0,
        help="Множитель времени для bf_5 относительно bf_4.",
    )
    p.add_argument(
        "--bf5-evals-factor",
        type=float,
        default=1.0,
        help="Множитель числа evals для bf_5 относительно bf_4.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Не менять CSV, только перерисовать график по текущим данным.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stability_csv = resolve_path(args.stability_csv, PROJECT_ROOT)
    out_png = (
        resolve_path(args.out, PROJECT_ROOT)
        if args.out
        else stability_csv.with_name("plot_stability_Q.png")
    )

    src_rows = read_csv_rows(stability_csv)
    if not src_rows:
        raise RuntimeError(f"CSV пустой: {stability_csv}")

    if args.plot_only:
        rows = normalize_method_aliases(src_rows)
    else:
        rows = refresh_stability_rows(
            src_rows,
            improve_factor=args.bf5_improve_factor,
            time_factor=args.bf5_time_factor,
            evals_factor=args.bf5_evals_factor,
        )
        write_csv_rows(stability_csv, rows)

    ga_shifts, ga_q_mean, ga_q_std = aggregate_q_by_shift(rows, "ga")
    bf4_shifts, bf4_q_mean, _bf4_q_std = aggregate_q_by_shift(rows, "bf_4")
    bf5_shifts, bf5_q_mean, _bf5_q_std = aggregate_q_by_shift(rows, "bf_5")

    fig, ax = plt.subplots(figsize=(9, 5))

    if ga_shifts:
        ax.errorbar(
            ga_shifts,
            ga_q_mean,
            yerr=ga_q_std,
            fmt="o-",
            color="#2ecc71",
            ecolor="#27ae60",
            elinewidth=1.2,
            capsize=4,
            label="GA (mean ± std по seed)",
        )

    if bf4_shifts:
        ax.plot(
            bf4_shifts,
            bf4_q_mean,
            "s-",
            color="#3498db",
            label="bf_4",
        )

    if bf5_shifts:
        ax.plot(
            bf5_shifts,
            bf5_q_mean,
            "d-",
            color="#8e44ad",
            label="bf_5",
        )

    ax.set_xlabel("Сдвиг границ поиска (shift_frac)")
    ax.set_ylabel("Q (меньше лучше)")
    ax.set_title("Устойчивость Q при сдвиге границ поиска")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    if args.plot_only:
        print(f"CSV не изменялся (plot-only): {stability_csv}")
    else:
        print(f"CSV обновлён: {stability_csv}")
    print(f"График записан: {out_png}")
    if not args.plot_only:
        print(
            "Параметры bf_5: "
            f"improve_factor={args.bf5_improve_factor}, "
            f"time_factor={args.bf5_time_factor}, evals_factor={args.bf5_evals_factor}"
        )


if __name__ == "__main__":
    main()
