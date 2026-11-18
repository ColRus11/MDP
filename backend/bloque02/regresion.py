"""Bloque 2: Regresión del índice de riesgo de infarto.

Estima un puntaje CONTINUO de riesgo de infarto (0 a 1) a partir de las variables
clínicas. A diferencia de la clasificación (sí/no) del Bloque 1, la regresión da un
valor continuo que permite PRIORIZAR pacientes y, sobre todo, CALIBRAR la recompensa
del MDP: cuanto mayor es el riesgo estimado, más conviene la acción ESCALAR.

Usa la misma limpieza del proyecto (`bloque01.datos.limpiar_datos`, 1.314 pacientes).
Compara varios regresores y reporta R², MAE y RMSE sobre un conjunto de prueba.

Ejecutar (desde la carpeta backend/):
    python -m bloque02.regresion
"""
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bloque01.datos import limpiar_datos

SEMILLA = 42
VARIABLES = [
    "Age", "Gender", "Heart rate", "Systolic blood pressure",
    "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin",
]
DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def preparar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """X = variables clínicas, y = riesgo de infarto (1.0 positivo, 0.0 negativo)."""
    df = limpiar_datos()
    X = df[VARIABLES].copy()
    y = (df["Result"] == "positive").astype(float)
    return X, y


def _construir_modelos() -> dict:
    return {
        "Lineal": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEMILLA),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEMILLA),
    }


@lru_cache(maxsize=1)
def entrenar_regresion() -> dict:
    """Entrena y evalúa los regresores. Devuelve métricas, importancias y el mejor modelo."""
    X, y = preparar_datos()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=(y > 0.5)
    )

    resultados = []
    modelos_entrenados = {}
    for nombre, modelo in _construir_modelos().items():
        modelo.fit(X_tr, y_tr)
        pred = np.clip(modelo.predict(X_te), 0.0, 1.0)   # riesgo acotado a [0, 1]
        resultados.append({
            "modelo": nombre,
            "r2": float(r2_score(y_te, pred)),
            "mae": float(mean_absolute_error(y_te, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        })
        modelos_entrenados[nombre] = modelo

    resultados.sort(key=lambda r: r["r2"], reverse=True)
    mejor = resultados[0]["modelo"]

    # Importancia de variables (del mejor modelo basado en árboles, si aplica)
    importancias = []
    modelo_imp = modelos_entrenados.get("Random Forest")
    if modelo_imp is not None:
        for var, imp in sorted(
            zip(VARIABLES, modelo_imp.feature_importances_),
            key=lambda t: t[1], reverse=True,
        ):
            importancias.append({"variable": var, "importancia": float(imp)})

    return {
        "resultados": resultados,
        "mejor_modelo": mejor,
        "importancias": importancias,
        "modelos": modelos_entrenados,
        "test": (X_te, y_te),
        "n_test": int(len(y_te)),
    }


def mapa_riesgo_accion(riesgo: float) -> str:
    """Puente regresión → MDP: convierte el riesgo continuo en el semáforo de triaje."""
    if riesgo >= 0.66:
        return "ESCALAR"       # rojo
    if riesgo >= 0.33:
        return "REPETIR"       # amarillo
    return "SEGUIMIENTO"       # verde


def exportar_resultados() -> dict:
    datos = entrenar_regresion()
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # --- JSON de resultados (para el frontend / informe) ---
    payload = {
        "resultados": datos["resultados"],
        "mejor_modelo": datos["mejor_modelo"],
        "importancias": datos["importancias"],
        "n_test": datos["n_test"],
        "objetivo": "riesgo_infarto_continuo_0_1",
    }
    (DIR_RESULTADOS / "regresion_resultados.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Gráfica 1: comparación de modelos (R² y MAE) ---
    res = datos["resultados"]
    nombres = [r["modelo"] for r in res]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.barh(nombres, [r["r2"] for r in res], color="#38BDF8")
    ax1.set_title("R² por modelo (mayor es mejor)"); ax1.set_xlim(0, 1); ax1.invert_yaxis()
    ax2.barh(nombres, [r["mae"] for r in res], color="#FF9F43")
    ax2.set_title("MAE por modelo (menor es mejor)"); ax2.invert_yaxis()
    fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "regresion_comparacion.png", dpi=150); plt.close(fig)

    # --- Gráfica 2: riesgo predicho vs real (mejor modelo) ---
    X_te, y_te = datos["test"]
    mejor_modelo = datos["modelos"][datos["mejor_modelo"]]
    pred = np.clip(mejor_modelo.predict(X_te), 0, 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    jitter = (np.random.default_rng(SEMILLA).random(len(y_te)) - 0.5) * 0.08
    ax.scatter(y_te + jitter, pred, s=12, alpha=0.4, color="#2DD4A7")
    ax.plot([0, 1], [0, 1], "--", color="#6E84A3", linewidth=1)
    for umbral in (0.33, 0.66):
        ax.axhline(umbral, color="#FF4D5E", alpha=0.25, linewidth=1)
    ax.set_xlabel("Resultado real (0 = sano, 1 = infarto)")
    ax.set_ylabel("Riesgo estimado (regresión)")
    ax.set_title(f"Riesgo predicho vs real — {datos['mejor_modelo']}")
    fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "regresion_pred_vs_real.png", dpi=150); plt.close(fig)

    return payload


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # consola Windows (cp1252) sin errores
    except Exception:
        pass
    salida = exportar_resultados()
    print("=== REGRESIÓN DE RIESGO DE INFARTO (Bloque 2) ===")
    print(f"Variables: {len(VARIABLES)} | Conjunto de prueba: {salida['n_test']} pacientes\n")
    print(f"{'modelo':>18} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    for r in salida["resultados"]:
        print(f"{r['modelo']:>18} {r['r2']:>8.4f} {r['mae']:>8.4f} {r['rmse']:>8.4f}")
    print(f"\nMejor modelo: {salida['mejor_modelo']}")
    print("\nImportancia de variables (Random Forest):")
    for imp in salida["importancias"]:
        print(f"  {imp['variable']:>26}  {imp['importancia']:.3f}")
    print("\nPuente regresión → MDP (riesgo → acción de triaje):")
    for r in (0.1, 0.5, 0.8):
        print(f"  riesgo={r:.2f}  →  {mapa_riesgo_accion(r)}")
    print(f"\nGuardados en {DIR_RESULTADOS}:")
    print("  regresion_resultados.json, regresion_comparacion.png, regresion_pred_vs_real.png")
