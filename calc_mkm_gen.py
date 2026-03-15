from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import lasio as ls
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

SINGULAR_PENALTY = 1.0e6


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_data_from_las(
    las_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lasdata = ls.read(las_path)

    data = lasdata.stack_curves(
        ["DEPT", "LITO", "POTA", "THOR", "RHOB", "TRNP"],
        sort_curves=False,
    )

    data = np.c_[data, np.ones(data.shape[0])]
    data[data[:, 1] != 1, 1] = 2

    is_coll = data[:, 1] == 1
    is_glin = data[:, 1] == 2

    coll_prop = data[is_coll][:, 2:]
    glin_prop = data[is_glin][:, 2:]

    return data, is_coll, is_glin, coll_prop, glin_prop


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


@dataclass
class GroupGAResult:
    best_matrix: np.ndarray
    best_score: float
    neg_share: float
    bad_share: float
    generations_ran: int


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
) -> tuple[list, tools.Logbook]:
    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(population)
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

    return population, logbook


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
            _, logbook = run_ea_with_patience(
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
        _, logbook = run_ea_with_patience(
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

    return GroupGAResult(
        best_matrix=best_matrix,
        best_score=best_score,
        neg_share=neg_share,
        bad_share=bad_share,
        generations_ran=generations_ran,
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
    coll_ratio = len(coll_prop) / (len(coll_prop) + len(glin_prop))
    glin_ratio = len(glin_prop) / (len(coll_prop) + len(glin_prop))

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
    search_time_sec = time.perf_counter() - search_start

    return coll_result, glin_result, search_time_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Генетический поиск матриц A_coll/A_glin для МКМ модели, "
            "печать статистики по поколениям и итоговых метрик."
        )
    )
    parser.add_argument("--las", default="inp.las", help="Путь к .las файлу.")
    parser.add_argument("--a-min-coll", default="a_min_coll.in", help="Путь к a_min_coll.in.")
    parser.add_argument("--a-max-coll", default="a_max_coll.in", help="Путь к a_max_coll.in.")
    parser.add_argument("--a-min-glin", default="a_min_glin.in", help="Путь к a_min_glin.in.")
    parser.add_argument("--a-max-glin", default="a_max_glin.in", help="Путь к a_max_glin.in.")

    parser.add_argument(
        "--best-coll-out",
        default="best_matrix_coll_gen.out",
        help="Куда сохранить лучшую матрицу коллектора.",
    )
    parser.add_argument(
        "--best-glin-out",
        default="best_matrix_glin_gen.out",
        help="Куда сохранить лучшую матрицу глины.",
    )
    parser.add_argument(
        "--plot-png",
        default="mkm_gen_best_plot.png",
        help="Путь для сохранения графика лучшей МКМ-модели в .png.",
    )
    parser.add_argument(
        "--save-mkm",
        default="",
        help="Необязательно: путь для сохранения лучшей МКМ-модели в .npy.",
    )

    parser.add_argument("--w-negative", type=float, default=0.7, help="Вес метрики отрицательных значений.")
    parser.add_argument("--w-glin", type=float, default=0.3, help="Вес метрики доли плохих глин.")
    parser.add_argument("--w-coll", type=float, default=0.3, help="Вес метрики доли плохих коллекторов.")

    parser.add_argument("--population-size", type=int, default=220, help="Размер популяции.")
    parser.add_argument("--ngen", type=int, default=110, help="Максимальное число поколений.")
    parser.add_argument("--cxpb", type=float, default=0.6, help="Вероятность кроссовера.")
    parser.add_argument("--mutpb", type=float, default=0.25, help="Вероятность мутации особи.")
    parser.add_argument("--indpb", type=float, default=0.1, help="Вероятность мутации гена.")
    parser.add_argument("--tournsize", type=int, default=3, help="Размер турнира при отборе.")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping по поколениями без улучшений.")
    parser.add_argument("--min-delta", type=float, default=1e-7, help="Минимальное улучшение для сброса patience.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Параллельные процессы для fitness (1 = без параллели).")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости.")
    return parser.parse_args()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    args = parse_args()

    las_path = resolve_path(args.las, script_dir)
    a_min_coll_path = resolve_path(args.a_min_coll, script_dir)
    a_max_coll_path = resolve_path(args.a_max_coll, script_dir)
    a_min_glin_path = resolve_path(args.a_min_glin, script_dir)
    a_max_glin_path = resolve_path(args.a_max_glin, script_dir)

    data, is_coll, is_glin, coll_prop, glin_prop = load_data_from_las(las_path)
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

    print("Старт GA-оптимизации матриц.")
    print(
        f"Параметры GA: pop={ga_params.population_size}, ngen={ga_params.ngen}, "
        f"cxpb={ga_params.cxpb}, mutpb={ga_params.mutpb}, indpb={ga_params.indpb}, "
        f"tournsize={ga_params.tournsize}, patience={ga_params.patience}, n_jobs={ga_params.n_jobs}"
    )
    print(
        f"Целевая функция качества: Q={args.w_negative}*negative + "
        f"{args.w_glin}*glin_bad + {args.w_coll}*coll_bad"
    )

    coll_result, glin_result, search_time_sec = optimize_mkm_with_ga(
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_min_coll=a_min_coll,
        a_max_coll=a_max_coll,
        a_min_glin=a_min_glin,
        a_max_glin=a_max_glin,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
        ga_params=ga_params,
        verbose=True,
    )

    best_mkm_model = calc_mkm_model(
        data=data,
        is_coll=is_coll,
        is_glin=is_glin,
        coll_prop=coll_prop,
        glin_prop=glin_prop,
        a_coll=coll_result.best_matrix,
        a_glin=glin_result.best_matrix,
    )
    negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(best_mkm_model)
    quality_score = calc_quality_score(
        negative_share=negative_share,
        glin_bad_share=glin_bad_share,
        coll_bad_share=coll_bad_share,
        w_negative=args.w_negative,
        w_glin=args.w_glin,
        w_coll=args.w_coll,
    )

    best_coll_path = resolve_path(args.best_coll_out, script_dir)
    best_glin_path = resolve_path(args.best_glin_out, script_dir)
    best_coll_path.parent.mkdir(parents=True, exist_ok=True)
    best_glin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(best_coll_path, coll_result.best_matrix, fmt="%.12g")
    np.savetxt(best_glin_path, glin_result.best_matrix, fmt="%.12g")

    plot_path = resolve_path(args.plot_png, script_dir)
    save_mkm_plot(best_mkm_model, plot_path)

    if args.save_mkm:
        save_mkm_path = resolve_path(args.save_mkm, script_dir)
        np.save(save_mkm_path, best_mkm_model)
        print(f"Лучшая МКМ-модель сохранена в: {save_mkm_path}")

    print("\nGA-оптимизация завершена.")
    print(f"Время поиска (только GA): {search_time_sec:.3f} сек")
    print(
        f"Поколений отработано: COLL={coll_result.generations_ran}, "
        f"GLIN={glin_result.generations_ran}"
    )
    print(f"Локальный score COLL: {coll_result.best_score:.8f}")
    print(f"Локальный score GLIN: {glin_result.best_score:.8f}")
    print(f"Итоговый Q score: {quality_score:.8f}")

    print("\nЛучшая матрица COLL:")
    print(coll_result.best_matrix)
    print("\nЛучшая матрица GLIN:")
    print(glin_result.best_matrix)

    print("\nМетрики лучшей МКМ-модели:")
    print(f"1) Доля отрицательных значений: {negative_share:.8f}")
    print(f"2) Доля глин, где сумма глин < 30%: {glin_bad_share:.8f}")
    print(f"3) Доля коллекторов, где сумма глин > 30%: {coll_bad_share:.8f}")

    print(f"\nЛучшая матрица COLL сохранена в: {best_coll_path}")
    print(f"Лучшая матрица GLIN сохранена в: {best_glin_path}")
    print(f"График лучшей МКМ-модели сохранен в: {plot_path}")


if __name__ == "__main__":
    main()
