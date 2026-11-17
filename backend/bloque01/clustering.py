import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .datos import preparar_clustering

SEMILLA = 42

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

def perfilar_kmeans(k=5) -> pd.DataFrame:
    df, X, _ = preparar_clustering()
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    df = df.copy()
    df["cluster"] = modelo.fit_predict(X)
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
METODOS_ENLACE = ["ward", "complete", "average"]

def perfilar_jerarquico(k=3, metodo="ward") -> pd.DataFrame:
    df, X, _ = preparar_clustering()
    Z = linkage(X, method=metodo)
    df = df.copy()
    df["cluster"] = fcluster(Z, k, criterion="maxclust")
    
    objetivo = df.columns[-1]
    df["infarto"] = (df[objetivo] == 1).astype(int) # O el valor que corresponda en tu dataset

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