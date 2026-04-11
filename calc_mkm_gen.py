"""
Обратная совместимость: `import calc_mkm_gen` для старых скриптов и ноутбуков.

Запуск с CLI: предпочтительно `python scripts/mkm_ga.py` или `python mkm_run_ga.py`.
"""

from __future__ import annotations

from mkm_core import (
    calc_coll_metrics,
    calc_glin_metrics,
    calc_mkm_model,
    calc_metrics_mkm,
    calc_quality_score,
    flatten_bounds,
    load_mkm_from_las,
    plot_with_sign,
    resolve_path,
    save_mkm_plot,
    validate_matrix_shape,
)
from mkm_ga_engine import (
    GAParams,
    GroupGAResult,
    SINGULAR_PENALTY,
    build_toolbox,
    clip_individual,
    ensure_deap_classes,
    evaluate_coll_individual,
    evaluate_glin_individual,
    optimize_mkm_with_ga,
    run_ea_with_patience,
    run_group_ga,
)
from mkm_run_ga import main, parse_args

load_data_from_las = load_mkm_from_las

if __name__ == "__main__":
    main()
