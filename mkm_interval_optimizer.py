from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mkm_bruteforce_engine import brute_force_best_coll, brute_force_best_glin
from mkm_core import (
    LithotypeInterval,
    calc_metrics_mkm,
    calc_mkm_model_by_intervals,
    calc_quality_score,
)
from mkm_ga_engine import GAParams, GroupGenerationState, optimize_mkm_with_ga


@dataclass
class IntervalOptimizationResult:
    interval_id: int
    lithotype: int
    start_idx: int
    end_idx: int
    depth_start: float
    depth_end: float
    sample_count: int
    method: str
    best_matrix: np.ndarray
    local_score: float
    negative_share: float
    bad_share: float
    bad_metric_name: str
    elapsed_sec: float
    evals: int
    invalid_count: int = 0
    generations_ran: int = 0
    generation_states: list[GroupGenerationState] = field(default_factory=list)


@dataclass
class IntervalGenerationQualityPoint:
    generation: int
    quality_score: float
    negative_share: float
    glin_bad_share: float
    coll_bad_share: float
    tested_matrices: int


@dataclass
class IntervalOptimizationSummary:
    method: str
    interval_results: list[IntervalOptimizationResult]
    interval_matrices: dict[int, np.ndarray]
    mkm_model: np.ndarray
    negative_share: float
    glin_bad_share: float
    coll_bad_share: float
    quality_score: float
    total_time_sec: float
    total_evals: int
    total_invalid_count: int
    total_generations: int
    quality_curve: list[IntervalGenerationQualityPoint] = field(default_factory=list)


def coarsen_k_matrix(a_k: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1:
        return np.asarray(a_k, dtype=int).copy()
    base = np.asarray(a_k, dtype=float)
    coarsened = np.floor((base - 1.0) / float(factor)).astype(int) + 1
    return np.maximum(coarsened, 1)


def _interval_bad_metric_name(lithotype: int) -> str:
    return "coll_bad_share" if lithotype == 1 else "glin_bad_share"


def _run_single_interval_bruteforce(
    interval: LithotypeInterval,
    *,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_k_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    a_k_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    max_iterations: int | None = None,
    verbose: bool = False,
) -> IntervalOptimizationResult:
    started = time.perf_counter()
    if interval.lithotype == 1:
        best_matrix, best_score, best_neg, best_bad, evals, invalid_count = brute_force_best_coll(
            coll_prop=interval.prop,
            a_min=a_min_coll,
            a_max=a_max_coll,
            a_k=a_k_coll,
            neg_weight_scaled=w_negative,
            coll_weight=w_coll,
            global_start_index=0,
            global_total=max_iterations or 0,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    else:
        best_matrix, best_score, best_neg, best_bad, evals, invalid_count = brute_force_best_glin(
            glin_prop=interval.prop,
            a_min=a_min_glin,
            a_max=a_max_glin,
            a_k=a_k_glin,
            neg_weight_scaled=w_negative,
            glin_weight=w_glin,
            global_start_index=0,
            global_total=max_iterations or 0,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    return IntervalOptimizationResult(
        interval_id=interval.interval_id,
        lithotype=interval.lithotype,
        start_idx=interval.start_idx,
        end_idx=interval.end_idx,
        depth_start=interval.depth_start,
        depth_end=interval.depth_end,
        sample_count=interval.size,
        method="bf",
        best_matrix=best_matrix,
        local_score=float(best_score),
        negative_share=float(best_neg),
        bad_share=float(best_bad),
        bad_metric_name=_interval_bad_metric_name(interval.lithotype),
        elapsed_sec=time.perf_counter() - started,
        evals=int(evals),
        invalid_count=int(invalid_count),
    )


def run_interval_bruteforce(
    *,
    data: np.ndarray,
    intervals: list[LithotypeInterval],
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_k_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    a_k_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    max_iterations: int | None = None,
    verbose: bool = False,
) -> IntervalOptimizationSummary:
    interval_results: list[IntervalOptimizationResult] = []
    interval_matrices: dict[int, np.ndarray] = {}

    started = time.perf_counter()
    for interval in intervals:
        result = _run_single_interval_bruteforce(
            interval,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_k_coll=a_k_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            a_k_glin=a_k_glin,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        interval_results.append(result)
        interval_matrices[result.interval_id] = result.best_matrix

    mkm_model = calc_mkm_model_by_intervals(data, intervals, interval_matrices)
    negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(mkm_model)
    quality_score = calc_quality_score(
        negative_share=negative_share,
        glin_bad_share=glin_bad_share,
        coll_bad_share=coll_bad_share,
        w_negative=w_negative,
        w_glin=w_glin,
        w_coll=w_coll,
    )

    return IntervalOptimizationSummary(
        method="bf",
        interval_results=interval_results,
        interval_matrices=interval_matrices,
        mkm_model=mkm_model,
        negative_share=float(negative_share),
        glin_bad_share=float(glin_bad_share),
        coll_bad_share=float(coll_bad_share),
        quality_score=float(quality_score),
        total_time_sec=time.perf_counter() - started,
        total_evals=int(sum(item.evals for item in interval_results)),
        total_invalid_count=int(sum(item.invalid_count for item in interval_results)),
        total_generations=0,
    )


def _run_single_interval_ga(
    interval: LithotypeInterval,
    *,
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    ga_params: GAParams,
    verbose: bool = False,
) -> IntervalOptimizationResult:
    started = time.perf_counter()
    empty_prop = np.zeros((0, interval.prop.shape[1]), dtype=float)
    if interval.lithotype == 1:
        coll_result, _glin_result, _ = optimize_mkm_with_ga(
            coll_prop=interval.prop,
            glin_prop=empty_prop,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            ga_params=ga_params,
            verbose=verbose,
        )
        best_matrix = coll_result.best_matrix
        best_score = coll_result.best_score
        best_neg = coll_result.neg_share
        best_bad = coll_result.bad_share
        evals = coll_result.fitness_evals
        generations = coll_result.generations_ran
        generation_states = coll_result.generation_states
    else:
        _coll_result, glin_result, _ = optimize_mkm_with_ga(
            coll_prop=empty_prop,
            glin_prop=interval.prop,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            ga_params=ga_params,
            verbose=verbose,
        )
        best_matrix = glin_result.best_matrix
        best_score = glin_result.best_score
        best_neg = glin_result.neg_share
        best_bad = glin_result.bad_share
        evals = glin_result.fitness_evals
        generations = glin_result.generations_ran
        generation_states = glin_result.generation_states

    return IntervalOptimizationResult(
        interval_id=interval.interval_id,
        lithotype=interval.lithotype,
        start_idx=interval.start_idx,
        end_idx=interval.end_idx,
        depth_start=interval.depth_start,
        depth_end=interval.depth_end,
        sample_count=interval.size,
        method="ga",
        best_matrix=np.asarray(best_matrix, dtype=float),
        local_score=float(best_score),
        negative_share=float(best_neg),
        bad_share=float(best_bad),
        bad_metric_name=_interval_bad_metric_name(interval.lithotype),
        elapsed_sec=time.perf_counter() - started,
        evals=int(evals),
        generations_ran=int(generations),
        generation_states=generation_states,
    )


def _state_for_generation(
    generation_states: list[GroupGenerationState],
    generation: int,
) -> GroupGenerationState:
    if not generation_states:
        raise ValueError("Пустая история поколений для интервала GA.")
    selected = generation_states[0]
    for state in generation_states:
        if state.generation > generation:
            break
        selected = state
    return selected


def build_interval_ga_quality_curve(
    *,
    data: np.ndarray,
    intervals: list[LithotypeInterval],
    interval_results: list[IntervalOptimizationResult],
    w_negative: float,
    w_glin: float,
    w_coll: float,
) -> list[IntervalGenerationQualityPoint]:
    if not interval_results:
        return []

    states_by_interval = {
        result.interval_id: result.generation_states
        for result in interval_results
    }
    max_generation = max(
        states[-1].generation
        for states in states_by_interval.values()
        if states
    )

    quality_curve: list[IntervalGenerationQualityPoint] = []
    for generation in range(max_generation + 1):
        generation_matrices: dict[int, np.ndarray] = {}
        tested_matrices = 0
        for interval in intervals:
            states = states_by_interval.get(interval.interval_id, [])
            state = _state_for_generation(states, generation)
            generation_matrices[interval.interval_id] = state.best_matrix
            tested_matrices += int(state.cumulative_fitness_evals)

        generation_mkm = calc_mkm_model_by_intervals(data, intervals, generation_matrices)
        negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(generation_mkm)
        quality_score = calc_quality_score(
            negative_share=negative_share,
            glin_bad_share=glin_bad_share,
            coll_bad_share=coll_bad_share,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
        )
        quality_curve.append(
            IntervalGenerationQualityPoint(
                generation=generation,
                quality_score=float(quality_score),
                negative_share=float(negative_share),
                glin_bad_share=float(glin_bad_share),
                coll_bad_share=float(coll_bad_share),
                tested_matrices=int(tested_matrices),
            )
        )
    return quality_curve


def run_interval_ga(
    *,
    data: np.ndarray,
    intervals: list[LithotypeInterval],
    a_min_coll: np.ndarray,
    a_max_coll: np.ndarray,
    a_min_glin: np.ndarray,
    a_max_glin: np.ndarray,
    w_negative: float,
    w_glin: float,
    w_coll: float,
    ga_params: GAParams,
    verbose: bool = False,
) -> IntervalOptimizationSummary:
    interval_results: list[IntervalOptimizationResult] = []
    interval_matrices: dict[int, np.ndarray] = {}

    started = time.perf_counter()
    for interval in intervals:
        result = _run_single_interval_ga(
            interval,
            a_min_coll=a_min_coll,
            a_max_coll=a_max_coll,
            a_min_glin=a_min_glin,
            a_max_glin=a_max_glin,
            w_negative=w_negative,
            w_glin=w_glin,
            w_coll=w_coll,
            ga_params=ga_params,
            verbose=verbose,
        )
        interval_results.append(result)
        interval_matrices[result.interval_id] = result.best_matrix

    mkm_model = calc_mkm_model_by_intervals(data, intervals, interval_matrices)
    negative_share, glin_bad_share, coll_bad_share = calc_metrics_mkm(mkm_model)
    quality_score = calc_quality_score(
        negative_share=negative_share,
        glin_bad_share=glin_bad_share,
        coll_bad_share=coll_bad_share,
        w_negative=w_negative,
        w_glin=w_glin,
        w_coll=w_coll,
    )
    quality_curve = build_interval_ga_quality_curve(
        data=data,
        intervals=intervals,
        interval_results=interval_results,
        w_negative=w_negative,
        w_glin=w_glin,
        w_coll=w_coll,
    )

    return IntervalOptimizationSummary(
        method="ga",
        interval_results=interval_results,
        interval_matrices=interval_matrices,
        mkm_model=mkm_model,
        negative_share=float(negative_share),
        glin_bad_share=float(glin_bad_share),
        coll_bad_share=float(coll_bad_share),
        quality_score=float(quality_score),
        total_time_sec=time.perf_counter() - started,
        total_evals=int(sum(item.evals for item in interval_results)),
        total_invalid_count=0,
        total_generations=int(sum(item.generations_ran for item in interval_results)),
        quality_curve=quality_curve,
    )


def write_interval_results_csv(results: list[IntervalOptimizationResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "interval_id",
                "lithotype",
                "start_idx",
                "end_idx",
                "depth_start",
                "depth_end",
                "sample_count",
                "method",
                "local_score",
                "negative_share",
                "bad_metric_name",
                "bad_share",
                "elapsed_sec",
                "evals",
                "invalid_count",
                "generations_ran",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "interval_id": item.interval_id,
                    "lithotype": item.lithotype,
                    "start_idx": item.start_idx,
                    "end_idx": item.end_idx,
                    "depth_start": item.depth_start,
                    "depth_end": item.depth_end,
                    "sample_count": item.sample_count,
                    "method": item.method,
                    "local_score": item.local_score,
                    "negative_share": item.negative_share,
                    "bad_metric_name": item.bad_metric_name,
                    "bad_share": item.bad_share,
                    "elapsed_sec": item.elapsed_sec,
                    "evals": item.evals,
                    "invalid_count": item.invalid_count,
                    "generations_ran": item.generations_ran,
                }
            )


def save_interval_matrices_npz(summary: IntervalOptimizationSummary, npz_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"interval_{interval_id}": matrix
        for interval_id, matrix in summary.interval_matrices.items()
    }
    np.savez(npz_path, **payload)
