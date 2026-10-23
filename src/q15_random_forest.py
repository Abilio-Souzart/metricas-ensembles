"""Questão 15: Random Forest (PEJOTA, divisão 80/20)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .config import FEATURES, SEED, TEST_SIZE_Q15
from .data import load_pejota, make_split
from .models import random_forest
from .utils import header, save_figure, save_table


def curva_oob(split, ns=(10, 25, 50, 100, 200, 300, 500)) -> pd.DataFrame:
    """Erro OOB e AUC no teste conforme o número de árvores (warm_start reaproveita as árvores)."""
    imp = SimpleImputer(strategy="median").fit(split.X_tr)
    X_tr, X_te = imp.transform(split.X_tr), imp.transform(split.X_te)
    rf = RandomForestClassifier(oob_score=True, warm_start=True, random_state=SEED, n_jobs=-1)
    linhas = []
    import warnings
    warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
    for n in ns:
        rf.set_params(n_estimators=n).fit(X_tr, split.y_tr)
        linhas.append({"n_arvores": n, "erro_OOB": 1 - rf.oob_score_,
                       "AUC_teste": roc_auc_score(split.y_te, rf.predict_proba(X_te)[:, 1])})
    return pd.DataFrame(linhas).set_index("n_arvores")


def plot_importancias(gini: pd.Series, perm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g = gini.head(10)[::-1]
    axes[0].barh(g.index, g.values, color="#90A4AE")
    axes[0].set_title("Importância por impureza (Gini)\nusada no item (b)")
    p = perm.head(10)[::-1]
    axes[1].barh(p.index, p["média"], xerr=p["desvio"], color="#1565C0")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("Importância por permutação\n(queda de AUC no teste, 30 repetições)")
    fig.tight_layout()
    save_figure(fig, "q15b_importancias")


def plot_oob(curva: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curva.index, curva["erro_OOB"], "o-", color="#F44336", label="Erro OOB (treino)")
    ax.plot(curva.index, 1 - curva["AUC_teste"], "s--", color="#1565C0", label="1 − AUC no teste")
    ax.set_xscale("log")
    ax.set_xlabel("Número de árvores")
    ax.set_ylabel("Erro")
    ax.set_title("Random Forest: mais árvores estabilizam o erro, sem overfitting")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "q15_oob_por_numero_de_arvores")


def run() -> dict:
    header("QUESTÃO 15 - Random Forest (PEJOTA, 80/20)")
    df = load_pejota()
    split = make_split(df, TEST_SIZE_Q15)
    print(f"Treino: {len(split.y_tr)} | Teste: {len(split.y_te)}")

    # (a)
    rf = random_forest().fit(split.X_tr, split.y_tr)
    oob = 1 - rf.named_steps["modelo"].oob_score_
    prob = rf.predict_proba(split.X_te)[:, 1]
    auc = roc_auc_score(split.y_te, prob)
    acc_te = np.mean((prob >= 0.5) == split.y_te)
    print(f"(a) Erro OOB = {oob:.4f} (acurácia OOB = {1 - oob:.1%}) | AUC teste = {auc:.4f} | "
          f"acurácia teste = {acc_te:.1%}")

    # (b)
    gini = pd.Series(rf.named_steps["modelo"].feature_importances_, index=FEATURES
                     ).sort_values(ascending=False)
    pi = permutation_importance(rf, split.X_te, split.y_te, scoring="roc_auc",
                                n_repeats=30, random_state=SEED, n_jobs=-1)
    perm = pd.DataFrame({"média": pi.importances_mean, "desvio": pi.importances_std},
                        index=FEATURES).sort_values("média", ascending=False)
    niveis = df[FEATURES].nunique().rename("valores distintos")
    tab = gini.rename("Gini").to_frame().join(perm["média"].rename("permutação (AUC)")).join(niveis)
    save_table(tab.round(4), "q15b_importancias")
    print("\n(b) Top 5 por impureza (Gini):")
    print(gini.head(5).round(4).to_string())
    print("    Top 5 por permutação (queda de AUC no teste):")
    print(perm.head(5).round(4).to_string())
    print("    Valores distintos das variáveis do top 5 Gini:",
          niveis[gini.head(5).index].to_dict())
    plot_importancias(gini, perm)

    # Complemento: OOB x número de árvores
    curva = curva_oob(split)
    save_table(curva.round(4), "q15_oob_por_numero_de_arvores")
    print("\n    Erro OOB e AUC por número de árvores:")
    print(curva.round(4).to_string())
    plot_oob(curva)

    return {"df": df, "split": split, "rf": rf, "auc": auc, "oob": oob, "gini": gini, "perm": perm}
