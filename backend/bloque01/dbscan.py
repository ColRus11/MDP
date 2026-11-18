import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from functools import lru_cache
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .datos import preparar_clustering

FIGURAS = Path(__file__).resolve().parent.parent / "figuras"
EXPLICACION_DBSCAN = (
    "DBSCAN identifica observaciones que no pertenecen a regiones "
    "suficientemente densas del espacio de características. Los pacientes "
    "identificados como ruido son observaciones atípicas desde el punto de "
    "vista geométrico, no necesariamente errores de medición ni pacientes "
    "con una condición clínica determinada."
)


def distancias_k(X, k) -> np.ndarray:
    # Distancia de cada paciente a su k-ésimo vecino (él mismo cuenta como el 1.º)
    vecinos = NearestNeighbors(n_neighbors=k).fit(X)
    distancias, _ = vecinos.kneighbors(X)
    return np.sort(distancias[:, -1])


def guardar_k_distancias(X, valores_k=(8, 14), ruta=FIGURAS / "k_distancias.png"):
    plt.figure(figsize=(9, 5))
    for k in valores_k:
        d = distancias_k(X, k)
        plt.plot(d, label=f"min_samples = {k}")
        percentiles = {p: round(float(np.percentile(d, p)), 2) for p in (50, 90, 95, 98, 99)}
        print(f"k={k}: percentiles {percentiles}, máximo {d.max():.2f}")
    plt.xlabel("Pacientes ordenados por distancia")
    plt.ylabel("Distancia al k-ésimo vecino")
    plt.title("Método de k-distancias para elegir eps (DBSCAN)")
    plt.legend()
    plt.grid(alpha=0.3)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en {ruta}")
def barrido_dbscan(X, valores_ms=(8, 14), valores_eps=(1.6, 2.0, 2.4, 2.8)):
    filas = []
    for ms in valores_ms:
        for eps in valores_eps:
            etiquetas = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            en_grupo = etiquetas != -1          # -1 es la etiqueta de ruido
            grupos = etiquetas[en_grupo]
            n_clusters = len(np.unique(grupos))
            silhouette = None
            if n_clusters >= 2:
                silhouette = float(silhouette_score(X[en_grupo], grupos))
            filas.append({
                "min_samples": ms,
                "eps": eps,
                "clusters": n_clusters,
                "ruido": int((~en_grupo).sum()),
                "mayor_cluster": int(np.bincount(grupos).max()) if n_clusters else 0,
                "silhouette": silhouette,
            })
    return filas
def perfilar_ruido(
    df: pd.DataFrame,
    X: pd.DataFrame,
    eps: float = 2.0,
    min_samples: int = 14
) -> pd.DataFrame:
    """
    Perfila los pacientes identificados como ruido por DBSCAN.

    Compara las características clínicas de:
      - ruido: pacientes con etiqueta -1
      - resto: pacientes pertenecientes al grupo denso

    Args:
        df: DataFrame original con las variables clínicas.
        X: Matriz de características preparada para clustering.
        eps: Radio de vecindad de DBSCAN.
        min_samples: Número mínimo de vecinos.

    Returns:
        DataFrame con estadísticas descriptivas por grupo.
    """

    if len(df) != len(X):
        raise ValueError(
            "El DataFrame original y X deben tener el mismo número de filas."
        )

    if "Result" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'Result' en el dataset."
        )

    # Ejecutar DBSCAN con los parámetros seleccionados.
    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    etiquetas = modelo.fit_predict(X)

    # Trabajar sobre una copia para no modificar el dataset original.
    datos = df.copy().reset_index(drop=True)

    # Asignar etiquetas.
    datos["grupo"] = np.where(
        etiquetas == -1,
        "ruido",
        "resto"
    )

    # Normalizar la variable objetivo.
    resultado = datos["Result"].astype(str).str.strip().str.lower()

    datos["infarto"] = resultado.map({
        "positive": 1,
        "negative": 0
    })

    if datos["infarto"].isna().any():
        raise ValueError(
            "La columna Result contiene valores distintos de "
            "'positive' y 'negative'. Revise el dataset."
        )

    # Variables que se utilizarán en el perfil clínico.
    variables_numericas = [
        "Age",
        "Heart rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Blood sugar",
        "CK-MB",
        "Troponin"
    ]

    faltantes = [
        columna
        for columna in variables_numericas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan variables clínicas en el dataset: {faltantes}"
        )

    # Construcción del perfil.
    perfil = datos.groupby("grupo", sort=True).agg(
        pacientes=("grupo", "size"),
        edad_promedio=("Age", "mean"),
        pulso_promedio=("Heart rate", "mean"),
        sistolica_promedio=("Systolic blood pressure", "mean"),
        diastolica_promedio=("Diastolic blood pressure", "mean"),
        glucosa_promedio=("Blood sugar", "mean"),
        ckmb_mediana=("CK-MB", "median"),
        troponina_mediana=("Troponin", "median"),
        porcentaje_infarto=("infarto", "mean")
    )

    # Convertir proporción a porcentaje.
    perfil["porcentaje_infarto"] *= 100

    # Ordenar primero ruido y después resto.
    orden = ["ruido", "resto"]
    perfil = perfil.reindex(
        [grupo for grupo in orden if grupo in perfil.index]
    )

    return perfil.round(2)


@lru_cache(maxsize=32)
def analizar_pacientes_atipicos(
    eps: float = 2.0,
    min_samples: int = 14,
) -> dict:
    df, X, _ = preparar_clustering()
    perfil = perfilar_ruido(df, X, eps=eps, min_samples=min_samples)
    total = int(len(df))
    grupos = []
    for grupo, etiqueta in (("ruido", "Ruido"), ("resto", "Resto")):
        if grupo not in perfil.index:
            continue
        fila = perfil.loc[grupo]
        grupos.append({
            "grupo": grupo,
            "etiqueta": etiqueta,
            "pacientes": int(fila["pacientes"]),
            "edad_promedio": float(fila["edad_promedio"]),
            "pulso_promedio": float(fila["pulso_promedio"]),
            "sistolica_promedio": float(fila["sistolica_promedio"]),
            "diastolica_promedio": float(fila["diastolica_promedio"]),
            "glucosa_promedio": float(fila["glucosa_promedio"]),
            "ckmb_mediana": float(fila["ckmb_mediana"]),
            "troponina_mediana": float(fila["troponina_mediana"]),
            "porcentaje_positivos": float(fila["porcentaje_infarto"]),
        })

    por_grupo = {fila["grupo"]: fila for fila in grupos}
    graficos = [
        {
            "titulo": "CK-MB",
            "unidad": "ng/mL · mediana",
            "campo": "ckmb_mediana",
        },
        {
            "titulo": "Troponina",
            "unidad": "ng/mL · mediana",
            "campo": "troponina_mediana",
        },
        {
            "titulo": "Edad",
            "unidad": "años · promedio",
            "campo": "edad_promedio",
        },
        {
            "titulo": "Resultados positivos",
            "unidad": "% del grupo",
            "campo": "porcentaje_positivos",
        },
    ]
    for grafico in graficos:
        grafico["valores"] = [
            {
                "etiqueta": fila["etiqueta"],
                "valor": fila[grafico["campo"]],
            }
            for fila in grupos
        ]

    ruido = por_grupo.get("ruido")
    pacientes_ruido = ruido["pacientes"] if ruido else 0
    pacientes_densos = por_grupo.get("resto", {}).get("pacientes", 0)
    return {
        "parametros": {"eps": float(eps), "min_samples": int(min_samples)},
        "total_pacientes": total,
        "pacientes_ruido": int(pacientes_ruido),
        "porcentaje_atipicos": (
            float(round(pacientes_ruido / total * 100, 2)) if total else 0.0
        ),
        "pacientes_densos": int(pacientes_densos),
        "comparacion": grupos,
        "graficos": graficos,
        "explicacion": EXPLICACION_DBSCAN,
    }


def imprimir_perfil_ruido(perfil: pd.DataFrame) -> None:
    """
    Imprime el perfil clínico con formato legible.
    """

    print("\n" + "=" * 75)
    print("PERFIL CLÍNICO DE PACIENTES IDENTIFICADOS POR DBSCAN")
    print("=" * 75)

    print(perfil.to_string())

    print("\n" + "-" * 75)
    print("INTERPRETACIÓN DE LOS RESULTADOS")
    print("-" * 75)

    if "ruido" not in perfil.index or "resto" not in perfil.index:
        print(
            "No existen ambos grupos para realizar una comparación."
        )
        return

    ruido = perfil.loc["ruido"]
    resto = perfil.loc["resto"]

    print(
        f"\nPacientes atípicos: {int(ruido['pacientes'])}"
    )

    print(
        f"Pacientes del grupo denso: {int(resto['pacientes'])}"
    )

    print(
        "\nDiferencias entre ruido y resto:"
    )

    variables = [
        ("ckmb_mediana", "CK-MB"),
        ("troponina_mediana", "Troponina"),
        ("edad_promedio", "Edad"),
        ("glucosa_promedio", "Glucosa"),
        ("porcentaje_infarto", "Porcentaje de resultados positivos")
    ]

    for columna, nombre in variables:
        diferencia = ruido[columna] - resto[columna]

        print(
            f"{nombre:35s}: "
            f"Ruido={ruido[columna]:8.2f} | "
            f"Resto={resto[columna]:8.2f} | "
            f"Diferencia={diferencia:+8.2f}"
        )

if __name__ == "__main__":

    print("\nINICIANDO ANÁLISIS DBSCAN")

    # Reutilizar la preparación de datos existente.
    df, X, _ = preparar_clustering()

    # Parámetros seleccionados para el análisis de ruido.
    EPS = 2.0
    MIN_SAMPLES = 14

    print("\nParámetros utilizados:")
    print(f"eps: {EPS}")
    print(f"min_samples: {MIN_SAMPLES}")

    # Calcular perfil clínico.
    perfil = perfilar_ruido(
        df=df,
        X=X,
        eps=EPS,
        min_samples=MIN_SAMPLES
    )

    # Mostrar resultados.
    imprimir_perfil_ruido(perfil)

    # Exportar resultados para documentación y análisis.
    ruta_salida = "resultados_perfil_dbscan.csv"

    perfil.to_csv(
        ruta_salida,
        encoding="utf-8"
    )

    print(f"\nPerfil exportado correctamente: {ruta_salida}")