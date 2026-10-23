"""Questão 16: Gradient Boosting e comparação dos modelos (PEJOTA, 80/20)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from .config import FEATURES, SEED
from .metrics import metricas
from .models import arvore, gradient_boosting, logistica, random_forest
from .utils import header, save_figure, save_table


def plot_roc(split, probs: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(split.y_te, p)
        ax.plot(fpr, tpr, lw=2, ls="--" if "log" in nome else "-",
                label=f"{nome} ({roc_auc_score(split.y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("Taxa de falsos positivos (bons recusados)")
    ax.set_ylabel("Recall (maus detectados)")
    ax.set_title("(b) Curvas ROC no teste (AUC)")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "q16b_roc")


def curva_boosting(split, n_max=600, taxas=(0.1, 0.03)) -> pd.DataFrame:
    """AUC de validação cruzada (5 folds, só treino) por número de iterações."""
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    resultado = {}
    for lr in taxas:
        aucs = np.zeros((5, n_max))
        for i, (a, b) in enumerate(folds.split(split.X_tr, split.y_tr)):
            gb = gradient_boosting(n_estimators=n_max, learning_rate=lr)
            gb.fit(split.X_tr.iloc[a], split.y_tr.iloc[a])
            Xb = gb.named_steps["imputacao"].transform(split.X_tr.iloc[b])
            for j, p in enumerate(gb.named_steps["modelo"].staged_predict_proba(Xb)):
                aucs[i, j] = roc_auc_score(split.y_tr.iloc[b], p[:, 1])
        resultado[f"learning_rate = {lr}"] = aucs.mean(axis=0)
    return pd.DataFrame(resultado, index=np.arange(1, n_max + 1))


def plot_curva_boosting(curva: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in curva:
        ax.plot(curva.index, curva[col], lw=2, label=col)
        k = curva[col].idxmax()
        ax.plot(k, curva[col].max(), "o", ms=8, mec="k", color=ax.lines[-1].get_color())
    ax.axvline(200, color="grey", ls=":", label="200 iterações (notebook)")
    ax.set_xlabel("Número de árvores (iterações do boosting)")
    ax.set_ylabel("AUC (CV 5 folds, treino)")
    ax.set_title("Gradient Boosting: com taxa 0,1, mais iterações começam a piorar")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "q16_boosting_iteracoes")


def run(q15: dict) -> dict:
    header("QUESTÃO 16 - Gradient Boosting e comparação (PEJOTA, 80/20)")
    split = q15["split"]

    # (a)
    gb = gradient_boosting().fit(split.X_tr, split.y_tr)
    auc_gb = roc_auc_score(split.y_te, gb.predict_proba(split.X_te)[:, 1])
    print(f"(a) Gradient Boosting (200 árvores, taxa 0,1, profundidade 3): AUC teste = {auc_gb:.4f}")

    # (b)
    modelos = {
        "Regressão logística": logistica(),
        "Árvore de decisão (prof. 5)": arvore(),
        "Random Forest (200)": q15["rf"],
        "Gradient Boosting (200)": gb,
        "Regressão logística com log (extensão)": logistica(log=True),
    }
    probs, linhas = {}, []
    for nome, m in modelos.items():
        if nome not in ("Random Forest (200)", "Gradient Boosting (200)"):
            m.fit(split.X_tr, split.y_tr)
        probs[nome] = m.predict_proba(split.X_te)[:, 1]
        linhas.append({"modelo": nome, **metricas(split.y_te, probs[nome])})
    comp = pd.DataFrame(linhas).set_index("modelo")
    save_table(comp.round(4), "q16b_comparacao_teste")
    print("\n(b) Comparação no teste (300 empresas, corte 0,5):")
    print(comp[["AUC", "acurácia", "recall", "precisão", "F1"]].round(3).to_string())
    plot_roc(split, probs)

    # Complemento: validação cruzada repetida na base inteira
    X = pd.concat([split.X_tr, split.X_te])
    y = pd.concat([split.y_tr, split.y_te])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    linhas = []
    for nome, m in modelos.items():
        s = cross_val_score(clone(m), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        linhas.append({"modelo": nome, "AUC CV": s.mean(), "erro padrão": s.std(ddof=1) / np.sqrt(len(s))})
    cvtab = pd.DataFrame(linhas).set_index("modelo")
    save_table(cvtab.round(4), "q16_comparacao_cv")
    print("\n    AUC em validação cruzada (5 folds x 3 repetições, base inteira):")
    print(cvtab.round(3).to_string())

    # Complemento: número de iterações do boosting
    curva = curva_boosting(split)
    save_table(curva.iloc[[0, 49, 99, 199, 299, 399, 599]].round(4), "q16_boosting_iteracoes")
    for col in curva:
        print(f"    {col}: AUC CV máxima {curva[col].max():.3f} em {curva[col].idxmax()} iterações; "
              f"com 200 iterações: {curva[col].loc[200]:.3f}; com 600: {curva[col].loc[600]:.3f}")
    plot_curva_boosting(curva)
    return {"comp": comp, "cvtab": cvtab, "curva": curva}
