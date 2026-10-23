"""Testes do pré-processamento e de propriedades das métricas usadas na análise."""
import numpy as np
import pytest

from src.config import FEATURES, TEST_SIZE_Q15
from src.data import load_pejota, make_split
from src.metrics import metricas
from src.models import random_forest


@pytest.fixture(scope="module")
def df():
    return load_pejota()


def test_variaveis(df):
    assert "IDADE" in df.columns                     # espaço removido do nome
    assert len(FEATURES) == 18 and "COD_CLI" not in FEATURES


def test_imputacao_so_com_treino(df):
    s = make_split(df, TEST_SIZE_Q15)
    rf = random_forest(n_estimators=10, oob=False).fit(s.X_tr, s.y_tr)
    assert np.allclose(rf.named_steps["imputacao"].statistics_, s.X_tr.median())


def test_sempre_bom_tem_recall_zero():
    y = np.array([0] * 90 + [1] * 10)
    m = metricas(y, np.zeros(100))
    assert m["acurácia"] == pytest.approx(0.9)
    assert m["recall"] == 0 and m["F1"] == 0


def test_rmse_maior_ou_igual_mae():
    rng = np.random.default_rng(0)
    for _ in range(100):
        e = rng.standard_t(3, size=30)
        assert np.sqrt(np.mean(e ** 2)) >= np.mean(np.abs(e)) - 1e-12


def test_corte_menor_aumenta_recall(df):
    s = make_split(df, 0.3)
    from src.models import logistica
    prob = logistica().fit(s.X_tr, s.y_tr).predict_proba(s.X_te)[:, 1]
    recalls = [metricas(s.y_te, prob, t)["recall"] for t in (0.7, 0.5, 0.3)]
    assert recalls == sorted(recalls)
