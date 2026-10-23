"""Caminhos e parâmetros globais."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base_pejota.xls"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

SEED = 42
TEST_SIZE_Q13 = 0.30         # Questão 13 (70/30, como nas listas anteriores)
TEST_SIZE_Q15 = 0.20         # Questões 15 e 16 (80/20, como no enunciado)
TEST_SIZE_IRIS = 0.25
THRESHOLDS = [0.3, 0.5, 0.7]
N_ARVORES = 200
PROFUNDIDADE_ARVORE = 5      # árvore usada no notebook para comparação

# Mesmas 18 variáveis do notebook (UF e RAMO ficaram de fora)
FEATURES = ["IFIN1", "IFIN2", "IFIN3", "IFIN4", "IFIN5", "IFIN6", "IFIN7", "IFIN8",
            "IFIN9", "IFIN10", "FATLIQ", "REST_TIPO_1", "REST_TIPO_2", "REST_TIPO_3",
            "REST_TIPO_4", "REST_TIPO_5", "IDADE", "NUM_SOCIOS"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
