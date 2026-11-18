from functools import lru_cache

from fastapi import APIRouter, Query

from .clasificacion import (
    consultar_paciente,
    entrenar_random_forest,
    obtener_configuracion_modelo,
)
from .clustering import (
    asignar_cluster,
    evaluar_jerarquico,
    perfilar_jerarquico,
    perfilar_kmeans,
)
from .dbscan import analizar_pacientes_atipicos
from .datos import estadisticas_datos, preparar_clustering, resumen_datos
from .esquemas import (
    AsignacionClusterResponse,
    DatosPaciente,
    DatosPacienteCluster,
)

router = APIRouter(prefix="/api/b1", tags=["bloque01"])


@lru_cache(maxsize=1)
def _obtener_modelo():
    return entrenar_random_forest()["modelo"]


@router.get("/datos/resumen")
def datos_resumen():
    return resumen_datos()


@router.get("/datos/estadisticas")
def datos_estadisticas():
    return estadisticas_datos()


@router.post("/clasificacion/predecir")
def predecir_paciente(
    paciente: DatosPaciente,
    indice_arbol: int = Query(default=0, ge=0, le=99),
):
    resultado = consultar_paciente(
        _obtener_modelo(),
        paciente.model_dump(by_alias=True),
        indice_arbol=indice_arbol,
    )
    clase_predicha = resultado["clase"]
    return {
        "prediction": str(resultado["prediccion"]),
        "probability": resultado["probabilidades"][clase_predicha],
        "clase": clase_predicha,
        "probabilities": resultado["probabilidades"],
        "explicacion_arbol": resultado["explicacion_arbol"],
    }


@router.get("/clasificacion/resumen")
def resumen_clasificacion():
    resultados = entrenar_random_forest()
    importancias = resultados["feature_importances"]
    return {
        "accuracy": float(resultados["accuracy"]),
        "precision": float(resultados["precision"]),
        "recall": float(resultados["recall"]),
        "f1_score": float(resultados["f1_score"]),
        "test_samples": int(resultados["test_samples"]),
        "clases": ["negative", "positive"],
        "confusion_matrix": resultados["confusion_matrix"].tolist(),
        "classification_report": resultados["classification_report_dict"],
        "feature_importances": [
            {"variable": str(variable), "importance": float(valor)}
            for variable, valor in importancias.items()
        ],
        "n_estimators": len(resultados["modelo"].estimators_),
        "configuracion": obtener_configuracion_modelo(),
    }


@router.post(
    "/clustering/asignar",
    response_model=AsignacionClusterResponse,
)
def asignar_cluster_paciente(paciente: DatosPacienteCluster):
    return asignar_cluster(paciente.model_dump(by_alias=True))


@router.get("/dbscan/analisis")
def analisis_dbscan(
    eps: float = Query(default=2.0, gt=0, le=20),
    min_samples: int = Query(default=14, ge=2, le=5000),
):
    return analizar_pacientes_atipicos(eps=eps, min_samples=min_samples)


@router.get("/clustering/jerarquico")
def clustering_jerarquico(
    k_min: int = Query(default=2, ge=2, le=10),
    k_max: int = Query(default=10, ge=2, le=12),
):
    """Métricas de clustering jerárquico (silueta, cofenética, tamaño mínimo)."""
    _, X, _ = preparar_clustering()
    return {"resultados": evaluar_jerarquico(X, k_min=k_min, k_max=k_max)}


@router.get("/clustering/perfiles")
def clustering_perfiles(
    metodo: str = Query(default="kmeans"),
    k: int = Query(default=5, ge=2, le=10),
):
    """Perfil clínico por grupo (K-Means o jerárquico) para la tabla del frontend."""
    if metodo == "kmeans":
        perfil = perfilar_kmeans(k=k)
    else:
        perfil = perfilar_jerarquico(k=k, metodo=metodo)
    perfiles = [
        {
            "cluster": int(cluster_id),
            "pacientes": int(fila["pacientes"]),
            "edad": float(fila["edad"]),
            "pulso": float(fila["pulso"]),
            "sistolica": float(fila["sistolica"]),
            "diastolica": float(fila["diastolica"]),
            "glucosa": float(fila["glucosa"]),
            "ckmb_mediana": float(fila["ckmb_mediana"]),
            "troponina_mediana": float(fila["troponina_mediana"]),
            "pct_infarto": float(fila["pct_infarto"]),
        }
        for cluster_id, fila in perfil.iterrows()
    ]
    return {"perfiles": perfiles}