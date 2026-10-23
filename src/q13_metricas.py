"""Questão 13: métricas de classificação com regressão logística (PEJOTA)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import SEED, TEST_SIZE_Q13, THRESHOLDS
from .data import load_pejota, make_split
from .metrics import metricas
from .models import arvore, logistica
from .utils import header, save_figure, save_table


def plot_matriz(cm) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["BOM (0)", "MAU (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("(a) Regressão logística, corte 0,5")
    fig.tight_layout()
    save_figure(fig, "q13a_matriz_confusao")


def plot_thresholds(y, prob) -> None:
    grade = np.linspace(0.02, 0.98, 97)
    tab = pd.DataFrame([metricas(y, prob, t) for t in grade], index=grade)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(grade, tab["precisão"], lw=2, label="Precisão (MAU)")
    ax.plot(grade, tab["recall"], lw=2, label="Recall (MAU)")
    ax.plot(grade, tab["F1"], "--", lw=2, label="F1")
    ax.plot(grade, tab["acurácia"], ":", lw=2, color="k", label="Acurácia")
    for t in THRESHOLDS:
        ax.axvline(t, color="grey", ls=":", lw=1)
    ax.set_xlabel("Ponto de corte (classifica como MAU se P(MAU) ≥ corte)")
    ax.set_ylabel("Valor")
    ax.set_title("(c) Métricas em função do ponto de corte")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "q13c_metricas_por_corte")


def corte_com_especificidade(y, prob, especificidade_alvo: float) -> float:
    """Menor corte cuja especificidade é pelo menos a especificidade alvo."""
    for t in np.sort(np.unique(prob)):
        if metricas(y, prob, t)["especificidade"] >= especificidade_alvo:
            return float(t)
    return 1.0


def run() -> dict:
    header("QUESTÃO 13 - Métricas de classificação (regressão logística, PEJOTA)")
    df = load_pejota()
    split = make_split(df, TEST_SIZE_Q13)
    print(f"Treino: {len(split.y_tr)} | Teste: {len(split.y_te)} | 18 variáveis")

    # (a)
    logit = logistica().fit(split.X_tr, split.y_tr)
    prob = logit.predict_proba(split.X_te)[:, 1]
    m = metricas(split.y_te, prob)
    print("\n(a) Corte 0,5:")
    print(f"    VP = {m['VP']} | VN = {m['VN']} | FP = {m['FP']} | FN = {m['FN']}")
    print("    " + " | ".join(f"{k} = {m[k]:.3f}" for k in
                              ["acurácia", "precisão", "recall", "especificidade", "F1", "AUC"]))
    plot_matriz(confusion_matrix(split.y_te, (prob >= 0.5).astype(int)))

    # (c)
    thr = pd.DataFrame({t: metricas(split.y_te, prob, t) for t in THRESHOLDS}).T
    thr.index.name = "corte"
    save_table(thr.round(4), "q13c_metricas_por_corte")
    print("\n(c) Métricas por ponto de corte:")
    print(thr[["acurácia", "precisão", "recall", "especificidade", "F1", "FN", "FP"]]
          .round(3).to_string())
    plot_thresholds(split.y_te, prob)

    # (d) Logística x árvore
    arv = arvore().fit(split.X_tr, split.y_tr)
    prob_arv = arv.predict_proba(split.X_te)[:, 1]
    m_arv = metricas(split.y_te, prob_arv)
    t_eq = corte_com_especificidade(split.y_te, prob, m_arv["especificidade"])
    m_eq = metricas(split.y_te, prob, t_eq)
    comp = pd.DataFrame({
        "Logística (corte 0,5)": m,
        "Árvore prof. 5 (corte 0,5)": m_arv,
        f"Logística (corte {t_eq:.2f}, mesma especificidade da árvore)": m_eq,
    }).T
    save_table(comp.round(4), "q13d_logistica_vs_arvore")
    print("\n(d) Logística x árvore:")
    print(comp[["acurácia", "precisão", "recall", "especificidade", "F1", "AUC", "FN"]]
          .round(3).to_string())

    # (e) Demonstração: base com 90% de bons pagadores
    rng = np.random.default_rng(SEED)
    y_te = split.y_te.to_numpy()
    idx_bom = np.where(y_te == 0)[0]
    idx_mau = rng.choice(np.where(y_te == 1)[0], size=len(idx_bom) // 9, replace=False)
    idx = np.concatenate([idx_bom, idx_mau])
    y90, p90 = y_te[idx], prob[idx]
    sempre_bom = metricas(y90, np.zeros(len(idx)))
    sempre_bom["AUC"] = 0.5
    demo = pd.DataFrame({"Sempre BOM": sempre_bom, "Regressão logística": metricas(y90, p90)}).T
    save_table(demo.round(4), "q13e_base_desbalanceada")
    print(f"\n(e) Subconjunto do teste com {len(idx_bom)} BOM e {len(idx_mau)} MAU (90/10):")
    print(demo[["acurácia", "precisão", "recall", "F1", "AUC"]].round(3).to_string())

    return {"df": df, "split": split, "m": m, "thr": thr, "comp": comp, "demo": demo}
