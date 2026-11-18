import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .datos import VARIABLES_CLUSTERING, VARIABLES_LOG, preparar_clustering

SEMILLA = 42
NOMBRES_CLUSTER = {
    0: "Presión por debajo del promedio, marcadores normales",
    1: "Presión por encima del promedio, marcadores normales",
    2: "Troponina muy por encima del promedio",
    3: "CK-MB muy por encima del promedio",
    4: "Glucosa muy por encima del promedio",
}
TAMANOS_CLUSTER_ESPERADOS = {0: 471, 1: 321, 2: 240, 3: 166, 4: 116}
logger = logging.getLogger(__name__)

def indice_dunn(X, etiquetas) -> float:
    D = squareform(pdist(X))
    grupos = np.unique(etiquetas)
    diametro_max = 0.0
    separacion_min = np.inf
    for i in grupos:
        idx_i = np.where(etiquetas == i)[0]
        diametro_max = max (diametro_max, D[np.ix_(idx_i, idx_i)].max())
        for j in grupos:
            if j>i:
                idx_j = np.where(etiquetas==j)[0]
                separacion_min =min(separacion_min, D[np.ix_(idx_i, idx_j)].min())
    return float(separacion_min/diametro_max)

def evaluar_kmeans (X, k_min=2, k_max=10) -> list[dict]:
    resultados = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
        etiquetas = modelo.fit_predict(X)
        resultados.append({
            "k": k,
            "inercia": float(modelo.inertia_),
            "silhouette": float(silhouette_score(X, etiquetas)),
            "dunn": indice_dunn(X, etiquetas),
        })
    return resultados

def _crear_perfil_kmeans(df: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = etiquetas
    df["infarto"] = (df["Result"] == "positive").astype(int)

    perfil = df.groupby("cluster").agg(
        pacientes=("cluster", "size"),
        edad=("Age", "mean"),
        pulso=("Heart rate", "mean"),
        sistolica=("Systolic blood pressure", "mean"),
        diastolica=("Diastolic blood pressure", "mean"),
        glucosa=("Blood sugar", "mean"),
        ckmb_mediana=("CK-MB", "median"),
        troponina_mediana=("Troponin", "median"),
        pct_infarto=("infarto", "mean"),
        pct_gender_1=("Gender", "mean"),
    )
    perfil["pct_infarto"] *= 100
    perfil["pct_gender_1"] *= 100
    return perfil.round(2)


def perfilar_kmeans(k=5) -> pd.DataFrame:
    df, X, _ = preparar_clustering()
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    return _crear_perfil_kmeans(df, modelo.fit_predict(X))


@lru_cache(maxsize=1)
def obtener_modelo_kmeans_final() -> tuple[KMeans, StandardScaler, pd.DataFrame]:
    df, X, escalador = preparar_clustering()
    modelo = KMeans(n_clusters=5, n_init=10, random_state=SEMILLA)
    etiquetas = modelo.fit_predict(X)
    perfil = _crear_perfil_kmeans(df, etiquetas)
    tamanos = {
        int(cluster_id): int(cantidad)
        for cluster_id, cantidad in perfil["pacientes"].items()
    }

    if tamanos != TAMANOS_CLUSTER_ESPERADOS:
        logger.warning(
            "Los tamaños de los clústeres K-Means no coinciden con los esperados: "
            "obtenidos=%s, esperados=%s. Los identificadores pudieron reordenarse; "
            "no se reasignarán automáticamente.",
            tamanos,
            TAMANOS_CLUSTER_ESPERADOS,
        )

    return modelo, escalador, perfil


def asignar_cluster(paciente: Mapping[str, float]) -> dict[str, object]:
    modelo, escalador, perfil = obtener_modelo_kmeans_final()
    entrada = pd.DataFrame(
        [{variable: paciente[variable] for variable in VARIABLES_CLUSTERING}],
        columns=VARIABLES_CLUSTERING,
    )
    for variable in VARIABLES_LOG:
        entrada[variable] = np.log(entrada[variable])

    cluster_id = int(modelo.predict(escalador.transform(entrada))[0])
    grupo = perfil.loc[cluster_id]

    return {
        "cluster_id": cluster_id,
        "nombre": NOMBRES_CLUSTER[cluster_id],
        "pacientes_en_grupo": int(grupo["pacientes"]),
        "pct_infarto_en_grupo": float(grupo["pct_infarto"]),
        "perfil": {
            "edad": float(grupo["edad"]),
            "pulso": float(grupo["pulso"]),
            "sistolica": float(grupo["sistolica"]),
            "diastolica": float(grupo["diastolica"]),
            "glucosa": float(grupo["glucosa"]),
            "ckmb_mediana": float(grupo["ckmb_mediana"]),
            "troponina_mediana": float(grupo["troponina_mediana"]),
        },
    }
METODOS_ENLACE = ["ward", "complete", "average"]

def perfilar_jerarquico(k=3, metodo="ward") -> pd.DataFrame:
    df, X, _ = preparar_clustering()
    etiquetas = fcluster(linkage(X, method=metodo), k, criterion="maxclust")
    return _crear_perfil_kmeans(df, etiquetas)
    


def evaluar_jerarquico(X, k_min=2, k_max=10) -> list[dict]:
    distancias = pdist(X)
    resultados = []
    for metodo in METODOS_ENLACE:
        Z = linkage(X, method=metodo)
        corr_cofenetica, _ = cophenet(Z, distancias)
        for k in range(k_min, k_max + 1):
            etiquetas = fcluster(Z, k, criterion="maxclust")
            resultados.append({
                "metodo": metodo,
                "k": k,
                "cofenetica": float(corr_cofenetica),
                "silhouette": float(silhouette_score(X, etiquetas)),
                "tamano_min": int(np.bincount(etiquetas)[1:].min()),
            })
    return resultados


METODOS_ENLACE = ["ward", "complete", "average"]


def evaluar_jerarquico(X, k_min=2, k_max=10) -> list[dict]:
    distancias = pdist(X)
    resultados = []
    for metodo in METODOS_ENLACE:
        Z = linkage(X, method=metodo)
        corr_cofenetica, _ = cophenet(Z, distancias)
        for k in range(k_min, k_max + 1):
            etiquetas = fcluster(Z, k, criterion="maxclust")
            resultados.append({
                "metodo": metodo,
                "k": k,
                "cofenetica": float(corr_cofenetica),
                "silhouette": float(silhouette_score(X, etiquetas)),
                "tamano_min": int(np.bincount(etiquetas)[1:].min()),
            })
    return resultados


def guardar_dendrograma(X, metodo="ward", ruta="figuras/dendrograma_ward.png"):
    Z = linkage(X, method=metodo)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
    plt.title(f"Dendrograma (enlace {metodo})")
    plt.xlabel("Pacientes por rama (entre paréntesis)")
    plt.ylabel("Distancia de unión")
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
def exportar_evaluacion_jerarquica(ruta="backend/figuras/metricas_jerarquico.csv"):
    _, X, _ = preparar_clustering()
    resultados = evaluar_jerarquico(X)
    df_res = pd.DataFrame(resultados)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(ruta, index=False)
    print(f"Métricas jerárquicas guardadas en {ruta}")


if __name__ == "__main__":
    _, X, _ = preparar_clustering()
    
    print("=== 1. EVALUACIÓN DE CLUSTERING JERÁRQUICO ===")
    print(f"{'metodo':>9} {'k':>2} {'cofen.':>7} {'silhouette':>11} {'min_clus':>9}")
    for r in evaluar_jerarquico(X):
        print(f"{r['metodo']:>9} {r['k']:>2} {r['cofenetica']:>7.3f} "
              f"{r['silhouette']:>11.4f} {r['tamano_min']:>9}")
    
    print("\n=== 2. GUARDANDO DENDROGRAMA ===")
    guardar_dendrograma(X, "ward", "backend/figuras/dendrograma_ward.png")
    print("Dendrograma guardado en backend/figuras/dendrograma_ward.png")

    print("\n=== 3. PERFIL CLÍNICO (Jerárquico - Ward, k=3) ===")
    perfil_jer = perfilar_jerarquico(k=3, metodo="ward")
    print(perfil_jer)

    print("\n=== 4. PERFIL CLÍNICO (K-Means, k=5) ===")
    perfil_km = perfilar_kmeans(k=5)
    print(perfil_km)