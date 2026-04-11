# GenAlghorithm_GIS

Применение генетического алгоритма и полного перебора для построения минералогической компонентной модели (МКМ) по данным ГИС из LAS.

## Структура каталога

| Путь | Назначение |
|------|------------|
| `config/` | Границы и сетки для матриц (`a_min_*.in`, `a_max_*.in`, `a_k_*.in`), эталонные матрицы для лаб. режима |
| `data/las/` | Входные файлы скважин (`*.las`) |
| `outputs/plots/` | Графики компонент МКМ по глубине |
| `outputs/matrices/` | Найденные матрицы `*_coll_*.out`, `*_glin_*.out` |
| `outputs/experiments/` | Результаты сравнения BF/GA и исследований (`skv621_bf_ga`, др.) |
| `experiments/` | Скрипты `compare_bf_ga_skv621.py`, `plot_experiment_results.py`, `plot_tune_tradeoff.py` |
| `notebooks/` | Jupyter-ноутбуки (`ProjectM*.ipynb`) |
| `scripts/` | Удобные точки входа CLI |

Ядро кода: `mkm_core.py` (загрузка LAS, модель, метрики, графики), `mkm_ga_engine.py` (DEAP), `mkm_bruteforce_engine.py` (перебор), `mkm_run_ga.py` / `mkm_run_bruteforce.py` (аргументы командной строки).

## Запуск

Из корня `GenAlghorithm_GIS` (где лежит `mkm_core.py`):

**Генетический алгоритм**

```bash
python mkm_run_ga.py
```

**Полный перебор**

```bash
python scripts/mkm_bruteforce.py
```

По умолчанию используется `data/las/skv621.las` (константа `DEFAULT_LAS_RELPATH` в `mkm_core.py`); для другой скважины укажите `--las путь/к/файлу.las`. Границы матриц читаются из `config/`.

### Универсальность LAS

- Кривые глубины и литологии: `--depth DEPT`, `--litho LITO` (при других мнемониках укажите явно).
- Четыре свойства для признакового вектора (плюс столбец единиц → матрица 5×5):  
  `--props POTA THOR RHOB WNKT`  
  Если `--props` не задан, используются `POTA`, `THOR`, `RHOB` и первая доступная из типичного списка (`TRNP`, `WNKT`, `NPHI`, …).

### Прочие скрипты

- `python calc_mkm_lab.py` — МКМ по фиксированным матрицам из `config/matrix_*.out`.
- `python analyze_ga_hyperparams.py` — исследование гиперпараметров (вывод в `outputs/ga_hyperparam_study` по умолчанию; в CSV добавлены **Q МКМ** и компоненты метрик; графики `*_Q.png`; опция `--skip-indpb-sweep`). Отдельный каталог результатов:  
  `python analyze_ga_hyperparams.py --output-dir outputs/ga_hyperparam_study_skv621`
- `python tune_mkm_gen_hyperparams.py` — случайный поиск настроек GA.
- `python experiments/compare_bf_ga_skv621.py --all` — бенчмарк GA / полный BF / BF с бюджетом оценок + устойчивость к сдвигу границ; затем `python experiments/plot_experiment_results.py --input-dir outputs/experiments/skv621_bf_ga`.
- `python experiments/plot_tune_tradeoff.py --csv outputs/mkm_gen_tuning_report.csv` — scatter время vs Q по результатам тюнера.

**Веса по умолчанию** у `mkm_run_bruteforce.py` и `mkm_run_ga.py` согласованы: `w_negative=0.7`, `w_glin=0.3`, `w_coll=0.3`. У bruteforce есть флаг `--quiet` (без печати каждой итерации).

## Метрики качества МКМ

1. Доля отрицательных значений среди компонент модели.  
2. Доля интервалов глины, где сумма долей двух глиновых компонент &lt; 30%.  
3. Доля интервалов коллектора, где эта сумма &gt; 30%.

## Зависимости

Python 3, `numpy`, `lasio`, `matplotlib`, `deap` (для GA).
