"""Генетический алгоритм (DEAP) для подбора матриц A_coll / A_glin."""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from deap import base, creator, tools

from mkm_core import calc_coll_metrics, calc_glin_metrics, flatten_bounds

SINGULAR_PENALTY = 1.0e6


def clip_individual(individual: list[float], lower: np.ndarray, upper: np.ndarray) -> None:
    for i in range(len(individual)):
        if individual[i] < lower[i]:
            individual[i] = float(lower[i])
        elif individual[i] > upper[i]:
            individual[i] = float(upper[i])


def ensure_deap_classes() -> None:
    if not hasattr(creator, "FitnessMinMKM"):
        creator.create("FitnessMinMKM", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualMKM"):
        creator.create("IndividualMKM", list, fitness=creator.FitnessMinMKM)


def evaluate_coll_individual(
    individual: list[float],
    coll_prop: np.ndarray,
    neg_weight_scaled: float,
    coll_weight: float,
) -> tuple[float]:
    matrix = np.array(individual, dtype=float).reshape(5, 5)
    try:
        neg_share, coll_bad_share = calc_coll_metrics(matrix, coll_prop)
        score = neg_weight_scaled * neg_share + coll_weight * coll_bad_share
        return (float(score),)
    except np.linalg.LinAlgError:
        return (SINGULAR_PENALTY,)


def evaluate_glin_individual(
    individual: list[float],
    glin_prop: np.ndarray,
    neg_weight_scaled: float,
    glin_weight: float,
) -> tuple[float]:
    matrix = np.array(individual, dtype=float).reshape(5, 5)
    try:
        neg_share, glin_bad_share = calc_glin_metrics(matrix, glin_prop)
        score = neg_weight_scaled * neg_share + glin_weight * glin_bad_share
        return (float(score),)
    except np.linalg.LinAlgError:
        return (SINGULAR_PENALTY,)


@dataclass
class GAParams:
    population_size: int
    ngen: int
    cxpb: float
    mutpb: float
    indpb: float
    tournsize: int
    patience: int
    min_delta: float
    n_jobs: int
    seed: int | None


# Единые значения по умолчанию для CLI и интервальных скриптов (размер популяции и операторы GA).
DEFAULT_GA_POPULATION_SIZE: int = 400
DEFAULT_GA_CXPB: float = 0.8
DEFAULT_GA_MUTPB: float = 0.1
DEFAULT_GA_INDPB: float = 0.5
DEFAULT_GA_TOURNSIZE: int = 2
DEFAULT_GA_PATIENCE: int = 50


@dataclass
class GroupGenerationState:
    generation: int
    best_matrix: np.ndarray
    best_score: float
    neg_share: float
    bad_share: float
    cumulative_fitness_evals: int


@dataclass
class GroupGAResult:
    best_matrix: np.ndarray
    best_score: float
    neg_share: float
    bad_share: float
    generations_ran: int
    fitness_evals: int
    generation_states: list[GroupGenerationState] = field(default_factory=list)


def _empty_group_result() -> GroupGAResult:
    return GroupGAResult(
        best_matrix=np.eye(5, dtype=float),
        best_score=0.0,
        neg_share=0.0,
        bad_share=0.0,
        generations_ran=0,
        fitness_evals=0,
        generation_states=[],
    )


def build_toolbox(
    lower: np.ndarray,
    upper: np.ndarray,
    evaluate_fn,
    indpb: float,
    tournsize: int,
) -> base.Toolbox:
    ensure_deap_classes()
    toolbox = base.Toolbox()
    n_genes = len(lower)

    def generate_random_individual() -> list[float]:
        return [
            random.uniform(float(lower[i]), float(upper[i]))
            for i in range(n_genes)
        ]

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

    toolbox.register("individual", tools.initIterate, creator.IndividualMKM, generate_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fn)
    toolbox.register("mate", mate_bounded)
    toolbox.register("mutate", mutate_bounded)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    return toolbox


def run_ea_with_patience(
    toolbox: base.Toolbox,
    population_size: int,
    ngen: int,
    cxpb: float,
    mutpb: float,
    stats: tools.Statistics,
    hall_of_fame: tools.HallOfFame,
    verbose: bool,
    patience: int,
    min_delta: float,
) -> tuple[list, tools.Logbook, list[list[float]]]:
    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    best_individual_history: list[list[float]] = []

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(population)
    best_individual_history.append(list(hall_of_fame[0]))
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best_min = float(record["min"])
    stagnant_gens = 0

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        hall_of_fame.update(population)
        best_individual_history.append(list(hall_of_fame[0]))

        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        current_min = float(record["min"])
        if current_min < best_min - min_delta:
            best_min = current_min
            stagnant_gens = 0
        else:
            stagnant_gens += 1

        if patience > 0 and stagnant_gens >= patience:
            if verbose:
                print(f"Early stop: нет улучшений {patience} поколений подряд.")
            break

    return population, logbook, best_individual_history


def run_group_ga(
    group_name: str,
    prop: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    evaluate_fn,
    metric_fn,
    params: GAParams,
    verbose: bool = True,
    seed_shift: int = 0,
) -> GroupGAResult:
    lower, upper = flatten_bounds(a_min, a_max)
    if params.seed is not None:
        random.seed(params.seed + seed_shift)
        np.random.seed(params.seed + seed_shift)

    toolbox = build_toolbox(
        lower=lower,
        upper=upper,
        evaluate_fn=evaluate_fn,
        indpb=params.indpb,
        tournsize=params.tournsize,
    )

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hall_of_fame = tools.HallOfFame(1)

    if verbose:
        print(f"\n=== GA поиск для {group_name} ===")

    if params.n_jobs > 1:
        with mp.Pool(processes=params.n_jobs) as pool:
            toolbox.register("map", pool.map)
            _, logbook, best_individual_history = run_ea_with_patience(
                toolbox=toolbox,
                population_size=params.population_size,
                ngen=params.ngen,
                cxpb=params.cxpb,
                mutpb=params.mutpb,
                stats=stats,
                hall_of_fame=hall_of_fame,
                verbose=verbose,
                patience=params.patience,
                min_delta=params.min_delta,
            )
    else:
        _, logbook, best_individual_history = run_ea_with_patience(
            toolbox=toolbox,
            population_size=params.population_size,
            ngen=params.ngen,
            cxpb=params.cxpb,
            mutpb=params.mutpb,
            stats=stats,
            hall_of_fame=hall_of_fame,
            verbose=verbose,
            patience=params.patience,
            min_delta=params.min_delta,
        )

    best_individual = hall_of_fame[0]
    best_matrix = np.array(best_individual, dtype=float).reshape(5, 5)
    best_score = float(best_individual.fitness.values[0])
    neg_share, bad_share = metric_fn(best_matrix, prop)
    generations_ran = int(logbook[-1]["gen"])
    fitness_evals = int(sum(int(row["nevals"]) for row in logbook))
    if len(best_individual_history) != len(logbook):
        raise RuntimeError(
            "Неконсистентная история GA: число лучших особей по поколениям "
            "не совпадает с числом записей logbook."
        )

    generation_states: list[GroupGenerationState] = []
    cumulative_evals = 0
    for row, best_genes in zip(logbook, best_individual_history):
        generation = int(row["gen"])
        cumulative_evals += int(row["nevals"])
        generation_matrix = np.array(best_genes, dtype=float).reshape(5, 5)
        generation_score = float(evaluate_fn(best_genes)[0])
        generation_neg_share, generation_bad_share = metric_fn(generation_matrix, prop)
        generation_states.append(
            GroupGenerationState(
                generation=generation,
                best_matrix=generation_matrix,
                best_score=generation_score,
                neg_share=float(generation_neg_share),
                bad_share=float(generation_bad_share),
                cumulative_fitness_evals=int(cumulative_evals),
            )
        )

    return GroupGAResult(
        best_matrix=best_matrix,
        best_score=best_score,
        neg_share=neg_share,
        bad_share=bad_share,
        generations_ran=generations_ran,
        fitness_evals=fitness_evals,
        generation_states=generation_states,
    )


def optimize_mkm_with_ga(
    coll_prop: np.ndarray,
    glin_prop: np.ndarray,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    ga_params: GAParams,
    verbose: bool = True,
) -> tuple[GroupGAResult, GroupGAResult, float]:
    total_rows = len(coll_prop) + len(glin_prop)
    if total_rows == 0:
        raise ValueError("Для GA нет ни одной строки свойств.")

    coll_ratio = len(coll_prop) / total_rows
    glin_ratio = len(glin_prop) / total_rows

    eval_coll = partial(
        evaluate_coll_individual,
        coll_prop=coll_prop,
        neg_weight_scaled=w_negative * coll_ratio,
        coll_weight=w_coll,
    )
    eval_glin = partial(
        evaluate_glin_individual,
        glin_prop=glin_prop,
        neg_weight_scaled=w_negative * glin_ratio,
        glin_weight=w_glin,
    )

    search_start = time.perf_counter()
    if len(coll_prop) > 0:
        coll_result = run_group_ga(
            group_name="COLL",
            prop=coll_prop,
            a_min=a_min_coll,
            a_max=a_max_coll,
            evaluate_fn=eval_coll,
            metric_fn=calc_coll_metrics,
            params=ga_params,
            verbose=verbose,
            seed_shift=0,
        )
    else:
        coll_result = _empty_group_result()

    if len(glin_prop) > 0:
        glin_result = run_group_ga(
            group_name="GLIN",
            prop=glin_prop,
            a_min=a_min_glin,
            a_max=a_max_glin,
            evaluate_fn=eval_glin,
            metric_fn=calc_glin_metrics,
            params=ga_params,
            verbose=verbose,
            seed_shift=10_000,
        )
    else:
        glin_result = _empty_group_result()
    search_time_sec = time.perf_counter() - search_start

    return coll_result, glin_result, search_time_sec
