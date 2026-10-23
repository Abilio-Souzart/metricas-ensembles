"""Questão 14: métricas de regressão (Iris: versicolor e virginica)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import SEED, TEST_SIZE_IRIS
from .utils import header, save_figure, save_table


def run() -> dict:
    header("QUESTÃO 14 - Regressão linear na Iris (versicolor + virginica)")
    iris = load_iris(as_frame=True).frame
    iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    iris = iris[iris["target"].isin([1, 2])].reset_index(drop=True)
    X, y = iris[["petal_width"]], iris["petal_length"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE_IRIS, random_state=SEED)
    lr = LinearRegression().fit(X_tr, y_tr)
    pred = lr.predict(X_te)
    erros = y_te.to_numpy() - pred

    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    r2_tr = r2_score(y_tr, lr.predict(X_tr))
    rmse_base = np.sqrt(np.mean((y_te - y_tr.mean()) ** 2))
    tab = pd.Series({
        "n treino": len(y_tr), "n teste": len(y_te),
        "intercepto": lr.intercept_, "inclinação (cm de comprimento por cm de largura)": lr.coef_[0],
        "MAE": mae, "RMSE": rmse, "RMSE / MAE": rmse / mae, "R² teste": r2, "R² treino": r2_tr,
        "RMSE prevendo a média do treino": rmse_base,
        "maior erro absoluto": np.abs(erros).max(),
        "desvio padrão de petal_length (teste)": y_te.std(ddof=0),
    })
    save_table(tab.round(4).rename("valor").to_frame(), "q14_metricas")
    print(tab.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cores = iris["target"].map({1: "#2196F3", 2: "#4CAF50"})
    axes[0].scatter(iris["petal_width"], iris["petal_length"], c=cores, s=25, alpha=0.7)
    grade = np.linspace(iris["petal_width"].min(), iris["petal_width"].max(), 50)
    axes[0].plot(grade, lr.predict(pd.DataFrame({"petal_width": grade})), "r-", lw=2)
    axes[0].set_xlabel("petal_width (cm)")
    axes[0].set_ylabel("petal_length (cm)")
    axes[0].set_title("Reta ajustada (azul: versicolor, verde: virginica)")
    axes[1].scatter(y_te, pred, s=40, color="#455A64")
    lim = [min(y_te.min(), pred.min()) - 0.2, max(y_te.max(), pred.max()) + 0.2]
    axes[1].plot(lim, lim, "r--", lw=1)
    axes[1].set_xlabel("petal_length real (cm)")
    axes[1].set_ylabel("petal_length previsto (cm)")
    axes[1].set_title(f"Teste: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.2f}")
    fig.tight_layout()
    save_figure(fig, "q14_regressao_iris")
    return {"tab": tab}
