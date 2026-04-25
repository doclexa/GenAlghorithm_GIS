#!/usr/bin/env python3
"""Собирает Markdown-отчёт по результатам интервального сравнения BF и GA."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkm_core import resolve_path  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def build_table(rows: list[dict[str, str]]) -> str:
    header = "| Метод | Q | negative | glin_bad | coll_bad | Время, с | Оценки |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append(
            "| {method} | {Q} | {negative_share} | {glin_bad_share} | {coll_bad_share} | {time_sec} | {evals} |".format(
                method=row["method"],
                Q=fmt_float(float(row["Q"])),
                negative_share=fmt_float(float(row["negative_share"])),
                glin_bad_share=fmt_float(float(row["glin_bad_share"])),
                coll_bad_share=fmt_float(float(row["coll_bad_share"])),
                time_sec=f"{float(row['time_sec']):.3f}",
                evals=row["evals"],
            )
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/experiments/skv621_bf_ga")
    parser.add_argument(
        "--output-md",
        default="",
        help="Пусто → <input-dir>/report_interval_bf_vs_ga.md",
    )
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir, PROJECT_ROOT)
    bench_rows = read_csv(input_dir / "benchmark_skv621.csv")
    interval_rows = read_csv(input_dir / "interval_comparison_bf_matched_vs_ga.csv")
    if not bench_rows:
        raise FileNotFoundError(f"Не найден benchmark_skv621.csv в {input_dir}")

    output_md = (
        resolve_path(args.output_md, PROJECT_ROOT)
        if args.output_md
        else input_dir / "report_interval_bf_vs_ga.md"
    )

    ga_row = next(row for row in bench_rows if row["method"] == "ga_interval")
    bf_full_row = next(row for row in bench_rows if row["method"] == "bf_full_interval")
    coarse_rows = [row for row in bench_rows if row["method"].startswith("bf_coarse_")]
    matched_row = next(
        (row for row in bench_rows if row["method"] == "bf_budget_time_matched"),
        min(
            coarse_rows,
            key=lambda row: abs(float(row["time_sec"]) - float(ga_row["time_sec"])),
        ),
    )

    ga_time = float(ga_row["time_sec"])
    bf_full_time = float(bf_full_row["time_sec"])
    matched_time = float(matched_row["time_sec"])
    ga_q = float(ga_row["Q"])
    bf_full_q = float(bf_full_row["Q"])
    matched_q = float(matched_row["Q"])

    if ga_time > 0:
        full_time_ratio = bf_full_time / ga_time
        matched_time_ratio = matched_time / ga_time
    else:
        full_time_ratio = 0.0
        matched_time_ratio = 0.0

    if interval_rows:
        worst_interval = max(interval_rows, key=lambda row: float(row["delta_local_score"]))
        worst_interval_text = (
            f"Наибольшее локальное отставание BF от GA наблюдается на интервале "
            f"{worst_interval['interval_id']} ({worst_interval['depth_start']}..{worst_interval['depth_end']}), "
            f"где разница локального score составляет {float(worst_interval['delta_local_score']):.6f}."
        )
    else:
        worst_interval_text = "Покомпонентное сравнение интервалов не найдено."

    report_text = f"""# Отчёт по сравнению интервального brute force и генетического алгоритма

## Постановка
В этой версии проекта матрицы МКМ подбираются не глобально для всей скважины, а отдельно для каждого непрерывного интервала одинаковой литологии по глубине. После этого интервальные результаты собираются в итоговую МКМ по всей скважине.

Использованная итоговая метрика качества:

`Q = 0.8 * negative_share + 0.1 * glin_bad_share + 0.1 * coll_bad_share`

Чем меньше `Q`, тем лучше итоговая модель.

## Сводная таблица
{build_table(bench_rows)}

## Основные выводы
- Полный интервальный brute force отработал в `{full_time_ratio:.2f}` раза дольше, чем интервальный GA.
- При этом итоговое качество полного BF равно `{bf_full_q:.6f}`, а у GA `{ga_q:.6f}`.
- Наиболее близкий по времени BF-компаратор: `{matched_row["method"]}`. Его время составляет `{matched_time_ratio:.2f}` от времени GA, а качество равно `{matched_q:.6f}`.
- Если у matched BF значение `Q` выше, это означает, что при сопоставимом времени GA даёт более качественные кривые МКМ.
- {worst_interval_text}

## Интерпретация
Полный перебор остаётся эталоном по идее поиска по дискретной сетке, но при интервальной постановке он вынужден многократно повторять перебор для каждого литотип-интервала. Из-за этого суммарное время быстро растёт с числом интервалов. Генетический алгоритм также оптимизирует интервалы независимо, но тратит меньше оценок на поиск перспективной области и потому лучше масштабируется по времени.

Для честного сравнения по времени в эксперимент также включён BF с ограниченным бюджетом итераций на интервал и набор огрублённых сеток. Огрубление уменьшает число разбиений матрицы, сокращает время работы, но одновременно ухудшает качество, потому что сетка становится реже и оптимум по дискретным узлам смещается.

## Визуализации
- [Итоговое качество по методам](plot_benchmark_Q.png)
- [Время vs качество](plot_time_vs_quality.png)
- [Операции vs качество](plot_evals_vs_quality.png)
- [Где BF теряет качество относительно GA](plot_interval_delta_local_score.png)

## Данные эксперимента
- [Сводный benchmark CSV](benchmark_skv621.csv)
- [Покомпонентные интервалы GA](intervals_ga.csv)
- [Покомпонентные интервалы полного BF](intervals_bf_full.csv)
- [Покомпонентные интервалы BF с time-matched budget](intervals_bf_budget_time_matched.csv)
- [Сравнение matched BF vs GA по интервалам](interval_comparison_bf_matched_vs_ga.csv)
- [Сравнение full BF vs GA по интервалам](interval_comparison_bf_full_vs_ga.csv)

## Заключение
Для дипломной работы получена практическая демонстрация двух тезисов:

1. Полный перебор по сетке требует больше времени, чем генетический алгоритм, если матрицы подбираются отдельно на каждом литотип-интервале.
2. При попытке приблизить время brute force к времени GA за счёт огрубления сетки качество МКМ ухудшается, и генетический алгоритм даёт более выгодный компромисс между временем расчёта и качеством кривых.
"""

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_text, encoding="utf-8")
    print(f"Записано: {output_md}")


if __name__ == "__main__":
    main()
