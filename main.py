"""Executa a análise completa de métricas de avaliação e ensembles.

Uso:
    python main.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")

from src import q13_metricas, q14_regressao_iris, q15_random_forest, q16_boosting  # noqa: E402
from src.config import FIG_DIR, TABLE_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    q13_metricas.run()
    q14_regressao_iris.run()
    q15 = q15_random_forest.run()
    q16_boosting.run(q15)
    print(f"\nTabelas: {TABLE_DIR}\nFiguras: {FIG_DIR}")


if __name__ == "__main__":
    main()
