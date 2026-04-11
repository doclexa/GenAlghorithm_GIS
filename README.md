# GenAlghorithm_GIS

Применение генетического алгоритма и полного перебора для построения минералогической компонентной модели (МКМ) по данным ГИС из LAS.

## Структура каталога

| Путь | Назначение |
|------|------------|
| `config/` | Границы и сетки для матриц (`a_min_*.in`, `a_max_*.in`, `a_k_*.in`), эталонные матрицы для лаб. режима |
| `data/las/` | Входные файлы скважин (`*.las`) |
| `outputs/plots/` | Графики компонент МКМ по глубине |
| `outputs/matrices/` | Найденные матрицы `*_coll_*.out`, `*_glin_*.out` |
| `outputs/experiments/` | Результаты исследований гиперпараметров GA |
| `notebooks/` | Jupyter-ноутбуки (`ProjectM*.ipynb`) |
| `scripts/` | Удобные точки входа CLI |

Ядро кода: `mkm_core.py` (загрузка LAS, модель, метрики, графики), `mkm_ga_engine.py` (DEAP), `mkm_bruteforce_engine.py` (перебор), `mkm_run_ga.py` / `mkm_run_bruteforce.py` (аргументы командной строки).

## Запуск

Из корня `GenAlghorithm_GIS` (где лежит `mkm_core.py`):

**Генетический алгоритм**

```bash
python scripts/mkm_ga.py --las data/las/inp.las
# или
python mkm_run_ga.py --las data/las/skv621.las
```

**Полный перебор**

```bash
python scripts/mkm_bruteforce.py --las data/las/inp.las
```

По умолчанию `--las data/las/inp.las`, границы матриц читаются из `config/`.

### Универсальность LAS

- Кривые глубины и литологии: `--depth DEPT`, `--litho LITO` (при других мнемониках укажите явно).
- Четыре свойства для признакового вектора (плюс столбец единиц → матрица 5×5):  
  `--props POTA THOR RHOB WNKT`  
  Если `--props` не задан, используются `POTA`, `THOR`, `RHOB` и первая доступная из типичного списка (`TRNP`, `WNKT`, `NPHI`, …).

### Прочие скрипты

- `python calc_mkm_lab.py` — МКМ по фиксированным матрицам из `config/matrix_*.out`.
- `python analyze_ga_hyperparams.py` — исследование гиперпараметров (вывод в `outputs/ga_hyperparam_study` по умолчанию).
- `python tune_mkm_gen_hyperparams.py` — случайный поиск настроек GA.

Обратная совместимость: `import calc_mkm_gen` по-прежнему даёт `GAParams`, `optimize_mkm_with_ga`, `load_mkm_from_las` (как `load_data_from_las`), и т.д.

## Метрики качества МКМ

1. Доля отрицательных значений среди компонент модели.  
2. Доля интервалов глины, где сумма долей двух глиновых компонент &lt; 30%.  
3. Доля интервалов коллектора, где эта сумма &gt; 30%.

## Зависимости

Python 3, `numpy`, `lasio`, `matplotlib`, `deap` (для GA).
