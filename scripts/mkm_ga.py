#!/usr/bin/env python3
"""Точка входа: генетический поиск МКМ (добавляет корень проекта в sys.path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mkm_run_ga import main

if __name__ == "__main__":
    main()
