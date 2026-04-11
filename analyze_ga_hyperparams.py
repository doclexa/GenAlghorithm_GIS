from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

from mkm_core import (
    PROJECT_ROOT,
    flatten_bounds,
    load_mkm_from_las as load_data_from_las,
    resolve_path,
    validate_matrix_shape,
)

SINGULAR_PENALTY = 1.0e6


@dataclass
class GAStudyConfig:
    population_size: int
    cxpb: float
    mutpb: float
    tournsize: int
    indpb: float
    ngen: int
    seed: int
    n_jobs: int


@dataclass
class CurveResult:
    parameter_name: str
    parameter_value: float
    history_negative_share: list[float]
    elapsed_sec: float
    final_best_share: float
    global_best_share: float
    global_best_count: int


def parse_int_list(csv_values: str) -> list[int]:
    values = [item.strip() for item in csv_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Список целых значений пуст.")
    return [int(v) for v in values]


def parse_float_list(csv_values: str) -> list[float]:
    values = [item.strip() for item in csv_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Список вещественных значений пуст.")
    return [float(v) for v in values]


def ensure_deap_classes() -> None:
    if not hasattr(creator, "FitnessMinNegativeStudy"):
        creator.create("FitnessMinNegativeStudy", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualNegativeStudy"):
        creator.create("IndividualNegativeStudy", list, fitness=creator.FitnessMinNegativeStudy)


def clip_individual(individual: list[float], lower: np.ndarray, upper: np.ndarray) -> None:
    for i in range(len(individual)):
        if individual[i] < lower[i]:
            individual[i] = float(lower[i])
        elif individual[i] > upper[i]:
            individual[i] = float(upper[i])


def evaluate_negative_share_full_model(
    individual: list[float],
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
) -> tuple[float]:
    a_coll = np.array(individual[:25], dtype=float).reshape(5, 5)
    a_glin = np.array(individual[25:], dtype=float).reshape(5, 5)
    try:
        inv_coll = np.linalg.inv(a_coll)
        inv_glin = np.linalg.inv(a_glin)
    except np.linalg.LinAlgError:
        return (SINGULAR_PENALTY,)

    mkm_coll = coll_prop @ inv_coll
    mkm_glin = glin_prop @ inv_glin

    negative_count = np.sum(mkm_coll < 0) + np.sum(mkm_glin < 0)
    total_count = mkm_coll.size + mkm_glin.size
    negative_share = negative_count / total_count
    return (float(negative_share),)


def build_toolbox(
    lower: np.ndarray,
    upper: np.ndarray,
    evaluate_fn,
    indpb: float,
    tournsize: int,
) -> base.Toolbox:
    ensure_deap_classes()
    n_genes = len(lower)
    toolbox = base.Toolbox()

    def generate_random_individual() -> list[float]:
        return [random.uniform(float(lower[i]), float(upper[i])) for i in range(n_genes)]

    def mate_bounded(ind1, ind2):
        tools.cxTwoPoint(ind1, ind2)
        clip_individual(ind1, lower, upper)
        clip_individual(ind2, lower, upper)
        return ind1, ind2

    def mutate_bounded(individual):
        for i in range(n_genes):
            if random.random() < indpb:
                individual[i] = random.uniform(float(lower[i]), float(upper[i]))
        clip_individual(individual, lower, upper)
        return (individual,)

    toolbox.register("individual", tools.initIterate, creator.IndividualNegativeStudy, generate_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fn)
    toolbox.register("mate", mate_bounded)
    toolbox.register("mutate", mutate_bounded)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    return toolbox


def run_ga_collect_history(
    config: GAStudyConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
) -> tuple[list[float], float]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    eval_fn = partial(
        evaluate_negative_share_full_model,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
    )
    toolbox = build_toolbox(
        lower=lower,
        upper=upper,
        evaluate_fn=eval_fn,
        indpb=config.indpb,
        tournsize=config.tournsize,
    )

    def run_loop() -> list[float]:
        population = toolbox.population(n=config.population_size)
        history: list[float] = []

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        history.append(float(min(ind.fitness.values[0] for ind in population)))

        for _ in range(1, config.ngen + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.cxpb:
                    toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            for mutant in offspring:
                if random.random() < config.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            history.append(float(min(ind.fitness.values[0] for ind in population)))

        return history

    started = time.perf_counter()
    if config.n_jobs > 1:
        with mp.Pool(processes=config.n_jobs) as pool:
            toolbox.register("map", pool.map)
            history = run_loop()
    else:
        history = run_loop()
    elapsed = time.perf_counter() - started
    return history, elapsed


def run_sweep_for_parameter(
    parameter_name: str,
    parameter_values: list[float],
    base_config: GAStudyConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    total_component_values: int,
    seed_offset: int,
) -> list[CurveResult]:
    results: list[CurveResult] = []
    for idx, value in enumerate(parameter_values):
        run_seed = base_config.seed + seed_offset * 1_000 + idx
        if parameter_name == "population_size":
            run_config = replace(base_config, population_size=int(value), seed=run_seed)
        elif parameter_name == "cxpb":
            run_config = replace(base_config, cxpb=float(value), seed=run_seed)
        elif parameter_name == "mutpb":
            run_config = replace(base_config, mutpb=float(value), seed=run_seed)
        elif parameter_name == "tournsize":
            run_config = replace(base_config, tournsize=int(value), seed=run_seed)
        else:
            raise ValueError(f"Неизвестный параметр: {parameter_name}")

        history, elapsed = run_ga_collect_history(
            config=run_config,
            lower=lower,
            upper=upper,
            coll_prop=coll_prop,
            glin_prop=glin_prop,
        )
        final_share = float(history[-1])
        global_best = float(min(history))
        best_count = int(round(global_best * total_component_values))

        result = CurveResult(
            parameter_name=parameter_name,
            parameter_value=float(value),
            history_negative_share=history,
            elapsed_sec=elapsed,
            final_best_share=final_share,
            global_best_share=global_best,
            global_best_count=best_count,
        )
        results.append(result)

        print(
            f"[{parameter_name}] value={value} | time={elapsed:.3f}s | "
            f"global_best_share={global_best:.8f} (~{best_count} отриц. значений)"
        )
    return results


def save_parameter_plot(
    parameter_name: str,
    curves: list[CurveResult],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for curve in curves:
        generations = np.arange(len(curve.history_negative_share))
        label_value = (
            f"{int(curve.parameter_value)}"
            if float(curve.parameter_value).is_integer()
            else f"{curve.parameter_value:.3g}"
        )
        ax.plot(generations, curve.history_negative_share, label=f"{parameter_name}={label_value}")

    ax.set_xlabel("Поколение")
    ax.set_ylabel("Доля отрицательных значений (лучшая особь)")
    ax.set_title(f"Влияние {parameter_name} на качество по поколениям")
    ax.grid(True)
    ax.legend(loc="best")

    speed_parts = []
    for curve in curves:
        value_label = (
            f"{int(curve.parameter_value)}"
            if float(curve.parameter_value).is_integer()
            else f"{curve.parameter_value:.3g}"
        )
        speed_parts.append(f"{parameter_name}={value_label}: {curve.elapsed_sec:.2f}s")
    speed_text = "Скорость 200 поколений: " + " | ".join(speed_parts)
    fig.text(0.5, 0.02, speed_text, ha="center", va="bottom", fontsize=10, wrap=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(path: Path, all_curves: list[CurveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "parameter",
                "value",
                "elapsed_sec",
                "final_best_negative_share",
                "global_best_negative_share",
                "global_best_negative_count",
            ]
        )
        for curve in all_curves:
            writer.writerow(
                [
                    curve.parameter_name,
                    curve.parameter_value,
                    f"{curve.elapsed_sec:.6f}",
                    f"{curve.final_best_share:.10f}",
                    f"{curve.global_best_share:.10f}",
                    curve.global_best_count,
                ]
            )


def save_curve_points_csv(path: Path, all_curves: list[CurveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "generation", "best_negative_share"])
        for curve in all_curves:
            for gen, share in enumerate(curve.history_negative_share):
                writer.writerow([curve.parameter_name, curve.parameter_value, gen, f"{share:.10f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Исследование влияния гиперпараметров GA на долю отрицательных значений "
            "и скорость работы на фиксированных 200 поколениях."
        )
    )
    parser.add_argument("--las", default="data/las/inp.las", help="Путь к .las (относительно корня проекта).")
    parser.add_argument("--a-min-coll", default="config/a_min_coll.in", help="Путь к a_min_coll.in.")
    parser.add_argument("--a-max-coll", default="config/a_max_coll.in", help="Путь к a_max_coll.in.")
    parser.add_argument("--a-min-glin", default="config/a_min_glin.in", help="Путь к a_min_glin.in.")
    parser.add_argument("--a-max-glin", default="config/a_max_glin.in", help="Путь к a_max_glin.in.")

    parser.add_argument("--ngen", type=int, default=200, help="Число поколений в каждом эксперименте.")
    parser.add_argument("--indpb", type=float, default=0.1, help="Вероятность мутации гена.")
    parser.add_argument("--seed", type=int, default=2026, help="Базовый seed.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Параллельные процессы для fitness (1 = без параллели).")

    parser.add_argument("--base-population-size", type=int, default=220, help="Базовый размер популяции.")
    parser.add_argument("--base-cxpb", type=float, default=0.6, help="Базовая вероятность скрещивания.")
    parser.add_argument("--base-mutpb", type=float, default=0.25, help="Базовая вероятность мутации особи.")
    parser.add_argument("--base-tournsize", type=int, default=3, help="Базовый tournament size.")

    parser.add_argument("--population-values", default="140,220,280,340", help="Список population_size через запятую.")
    parser.add_argument("--cxpb-values", default="0.5,0.6,0.7", help="Список cxpb через запятую.")
    parser.add_argument("--mutpb-values", default="0.2,0.25,0.3,0.35", help="Список mutpb через запятую.")
    parser.add_argument("--tournsize-values", default="2,3,4", help="Список tournsize через запятую.")

    parser.add_argument(
        "--output-dir",
        default="outputs/ga_hyperparam_study",
        help="Каталог для графиков и таблиц (относительно корня проекта).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    population_values = [float(v) for v in parse_int_list(args.population_values)]
    cxpb_values = parse_float_list(args.cxpb_values)
    mutpb_values = parse_float_list(args.mutpb_values)
    tournsize_values = [float(v) for v in parse_int_list(args.tournsize_values)]

    las_path = resolve_path(args.las, PROJECT_ROOT)
    a_min_coll_path = resolve_path(args.a_min_coll, PROJECT_ROOT)
    a_max_coll_path = resolve_path(args.a_max_coll, PROJECT_ROOT)
    a_min_glin_path = resolve_path(args.a_min_glin, PROJECT_ROOT)
    a_max_glin_path = resolve_path(args.a_max_glin, PROJECT_ROOT)

    _, _, _, coll_prop, glin_prop = load_data_from_las(las_path)
    if len(coll_prop) == 0:
        raise ValueError("В данных нет строк с LITO == 1 (коллекторы).")
    if len(glin_prop) == 0:
        raise ValueError("В данных нет строк с LITO != 1 (глины после преобразования в 2).")

    a_min_coll = np.loadtxt(a_min_coll_path)
    a_max_coll = np.loadtxt(a_max_coll_path)
    a_min_glin = np.loadtxt(a_min_glin_path)
    a_max_glin = np.loadtxt(a_max_glin_path)
    validate_matrix_shape(a_min_coll, "A_min_coll")
    validate_matrix_shape(a_max_coll, "A_max_coll")
    validate_matrix_shape(a_min_glin, "A_min_glin")
    validate_matrix_shape(a_max_glin, "A_max_glin")

    lower_coll, upper_coll = flatten_bounds(a_min_coll, a_max_coll)
    lower_glin, upper_glin = flatten_bounds(a_min_glin, a_max_glin)
    lower = np.concatenate([lower_coll, lower_glin])
    upper = np.concatenate([upper_coll, upper_glin])

    total_component_values = coll_prop.shape[0] * coll_prop.shape[1] + glin_prop.shape[0] * glin_prop.shape[1]

    base_config = GAStudyConfig(
        population_size=args.base_population_size,
        cxpb=args.base_cxpb,
        mutpb=args.base_mutpb,
        tournsize=args.base_tournsize,
        indpb=args.indpb,
        ngen=args.ngen,
        seed=args.seed,
        n_jobs=max(1, args.n_jobs),
    )

    print("Старт исследования влияния гиперпараметров GA.")
    print(
        f"Фиксированно: ngen={base_config.ngen}, indpb={base_config.indpb}, "
        f"base(pop={base_config.population_size}, cxpb={base_config.cxpb}, "
        f"mutpb={base_config.mutpb}, tournsize={base_config.tournsize})"
    )

    all_curves: list[CurveResult] = []

    population_curves = run_sweep_for_parameter(
        parameter_name="population_size",
        parameter_values=population_values,
        base_config=base_config,
        lower=lower,
        upper=upper,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        total_component_values=total_component_values,
        seed_offset=1,
    )
    all_curves.extend(population_curves)
    save_parameter_plot(
        parameter_name="population_size",
        curves=population_curves,
        output_path=output_dir / "ga_effect_population_size.png",
    )

    cxpb_curves = run_sweep_for_parameter(
        parameter_name="cxpb",
        parameter_values=cxpb_values,
        base_config=base_config,
        lower=lower,
        upper=upper,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        total_component_values=total_component_values,
        seed_offset=2,
    )
    all_curves.extend(cxpb_curves)
    save_parameter_plot(
        parameter_name="cxpb",
        curves=cxpb_curves,
        output_path=output_dir / "ga_effect_cxpb.png",
    )

    mutpb_curves = run_sweep_for_parameter(
        parameter_name="mutpb",
        parameter_values=mutpb_values,
        base_config=base_config,
        lower=lower,
        upper=upper,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        total_component_values=total_component_values,
        seed_offset=3,
    )
    all_curves.extend(mutpb_curves)
    save_parameter_plot(
        parameter_name="mutpb",
        curves=mutpb_curves,
        output_path=output_dir / "ga_effect_mutpb.png",
    )

    tournsize_curves = run_sweep_for_parameter(
        parameter_name="tournsize",
        parameter_values=tournsize_values,
        base_config=base_config,
        lower=lower,
        upper=upper,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        total_component_values=total_component_values,
        seed_offset=4,
    )
    all_curves.extend(tournsize_curves)
    save_parameter_plot(
        parameter_name="tournsize",
        curves=tournsize_curves,
        output_path=output_dir / "ga_effect_tournsize.png",
    )

    save_summary_csv(output_dir / "ga_hyperparam_study_summary.csv", all_curves)
    save_curve_points_csv(output_dir / "ga_hyperparam_study_curves.csv", all_curves)

    print("\nИсследование завершено.")
    print(f"Графики сохранены в: {output_dir}")
    print(f"- {output_dir / 'ga_effect_population_size.png'}")
    print(f"- {output_dir / 'ga_effect_cxpb.png'}")
    print(f"- {output_dir / 'ga_effect_mutpb.png'}")
    print(f"- {output_dir / 'ga_effect_tournsize.png'}")
    print(f"Сводная таблица: {output_dir / 'ga_hyperparam_study_summary.csv'}")
    print(f"Точки кривых: {output_dir / 'ga_hyperparam_study_curves.csv'}")


if __name__ == "__main__":
    main()
