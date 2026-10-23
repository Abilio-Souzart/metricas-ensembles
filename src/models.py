"""Modelos usados no projeto, todos como Pipeline.

A imputação pela mediana (e a padronização, quando usada) é ajustada só no
treino, ou só nos folds de treino da validação cruzada.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import N_ARVORES, PROFUNDIDADE_ARVORE, SEED


def signed_log(x):
    """sign(x) * log(1 + |x|) — mesma transformação dos projetos de logística e KNN."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _imputer():
    return ("imputacao", SimpleImputer(strategy="median"))


def logistica(log: bool = False) -> Pipeline:
    passos = [_imputer()]
    if log:
        passos.append(("log", FunctionTransformer(signed_log)))
    passos += [("escala", StandardScaler()),
               ("modelo", LogisticRegression(max_iter=1000, random_state=SEED))]
    return Pipeline(passos)


def arvore(max_depth: int = PROFUNDIDADE_ARVORE) -> Pipeline:
    return Pipeline([_imputer(), ("modelo", DecisionTreeClassifier(max_depth=max_depth,
                                                                   random_state=SEED))])


def random_forest(n_estimators: int = N_ARVORES, oob: bool = True, **kw) -> Pipeline:
    return Pipeline([_imputer(), ("modelo", RandomForestClassifier(
        n_estimators=n_estimators, oob_score=oob, random_state=SEED, n_jobs=-1, **kw))])


def gradient_boosting(n_estimators: int = N_ARVORES, learning_rate: float = 0.1,
                      max_depth: int = 3) -> Pipeline:
    return Pipeline([_imputer(), ("modelo", GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        random_state=SEED))])
