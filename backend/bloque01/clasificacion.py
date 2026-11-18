import hashlib
import logging
import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .datos import RUTA_CSV, limpiar_datos

SEMILLA = 42
VERSION_MODELO = 1
N_ESTIMATORS = 100
MAX_PROFUNDIDAD = None
TAMANO_PRUEBA = 0.2
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "random_forest.joblib"
_model_lock = Lock()
logger = logging.getLogger(__name__)

VARIABLES_PREDICTORAS = [
    "Age", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Blood sugar", "CK-MB", "Troponin",
]
VARIABLES_LOG = ["CK-MB", "Troponin"]

def preparar_datos_clasificacion():
    df = limpiar_datos()
    X = df[VARIABLES_PREDICTORAS].copy()
    
    # Aplicar transformación logarítmica a variables sesgadas (igual que en clustering)
    for c in VARIABLES_LOG:
        # Usamos np.log con un pequeño offset si hay ceros exactos
        X[c] = np.log(X[c] + 1e-5)
        
    # Variable objetivo (la última columna del dataset, ej: "Result")
    objetivo = df.columns[-1]
    y = df[objetivo]
    
    # Si la variable objetivo es texto (ej: "positive" / "negative"), la pasamos a binario (1 y 0)
    if y.dtype == 'object':
        y = (y.str.lower() == "positive").astype(int)
        
    return X, y

def _entrenar_random_forest():
    X, y = preparar_datos_clasificacion()
    
    # 1. División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_PRUEBA, random_state=SEMILLA, stratify=y
    )
    
    # 2. Instanciar y entrenar Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_PROFUNDIDAD,
        random_state=SEMILLA, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 3. Predicciones
    y_pred = rf.predict(X_test)
    
    # 4. Métricas de evaluación
    acc = accuracy_score(y_test, y_pred)
    matriz_conf = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    clase_positiva = "positive" if "positive" in rf.classes_ else 1
    etiquetas_reporte = [
        "negative" if clase_positiva == "positive" else 0,
        clase_positiva,
    ]
    reporte_detallado = classification_report(
        y_test,
        y_pred,
        labels=etiquetas_reporte,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    
    # 5. Importancia de las variables
    importancias = pd.Series(rf.feature_importances_, index=VARIABLES_PREDICTORAS).sort_values(ascending=False)
    
    resultados = {
        "accuracy": acc,
        "confusion_matrix": matriz_conf,
        "classification_report": reporte,
        "classification_report_dict": reporte_detallado,
        "precision": precision_score(
            y_test, y_pred, pos_label=clase_positiva, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, pos_label=clase_positiva, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, pos_label=clase_positiva, zero_division=0
        ),
        "test_samples": len(y_test),
        "feature_importances": importancias,
        "modelo": rf
    }
    
    return resultados


def _metadata_modelo() -> dict:
    return {
        "version": VERSION_MODELO,
        "dataset_sha256": hashlib.sha256(RUTA_CSV.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "configuracion": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_PROFUNDIDAD,
            "random_state": SEMILLA,
            "test_size": TAMANO_PRUEBA,
            "variables_predictoras": tuple(VARIABLES_PREDICTORAS),
            "variables_log": tuple(VARIABLES_LOG),
            "offset_log": 1e-5,
        },
    }


@lru_cache(maxsize=1)
def entrenar_random_forest():
    with _model_lock:
        metadata = _metadata_modelo()
        if RUTA_MODELO.exists():
            try:
                artefacto = joblib.load(RUTA_MODELO)
            except (
                AttributeError,
                EOFError,
                ImportError,
                OSError,
                pickle.UnpicklingError,
                ValueError,
            ) as error:
                logger.warning(
                    "No se pudo cargar el artefacto Random Forest (%s); "
                    "se entrenará una versión nueva.",
                    error,
                )
            else:
                resultados_guardados = (
                    artefacto.get("resultados")
                    if isinstance(artefacto, dict)
                    and artefacto.get("metadata") == metadata
                    else None
                )
                if (
                    isinstance(resultados_guardados, dict)
                    and isinstance(
                        resultados_guardados.get("modelo"),
                        RandomForestClassifier,
                    )
                ):
                    logger.info("Random Forest cargado desde %s", RUTA_MODELO)
                    return resultados_guardados
                logger.warning(
                    "El artefacto Random Forest no coincide con los datos o la "
                    "configuración vigente; se entrenará una versión nueva."
                )

        resultados = _entrenar_random_forest()
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=RUTA_MODELO.parent,
            suffix=".joblib.tmp",
            delete=False,
        ) as archivo_temporal:
            ruta_temporal = Path(archivo_temporal.name)
        try:
            joblib.dump(
                {"metadata": metadata, "resultados": resultados},
                ruta_temporal,
            )
            os.replace(ruta_temporal, RUTA_MODELO)
        finally:
            ruta_temporal.unlink(missing_ok=True)
        logger.info("Random Forest guardado en %s", RUTA_MODELO)
        return resultados


def obtener_configuracion_modelo() -> dict:
    metadata = _metadata_modelo()
    return {
        **metadata["configuracion"],
        "version": VERSION_MODELO,
        "algoritmo": "RandomForestClassifier",
        "sklearn_version": metadata["sklearn_version"],
        "dataset_sha256": metadata["dataset_sha256"],
    }


def preparar_paciente(datos: Mapping[str, float]) -> pd.DataFrame:
    paciente = pd.DataFrame(
        [{variable: datos[variable] for variable in VARIABLES_PREDICTORAS}],
        columns=VARIABLES_PREDICTORAS,
    )
    for variable in VARIABLES_LOG:
        paciente[variable] = np.log(paciente[variable] + 1e-5)
    return paciente


def _normalizar_clase(clase: object) -> str:
    etiqueta = str(clase).strip().lower()
    if etiqueta in {"1", "true", "positive"}:
        return "positive"
    if etiqueta in {"0", "false", "negative"}:
        return "negative"
    return etiqueta


def consultar_paciente(
    modelo: RandomForestClassifier,
    datos: Mapping[str, float],
    indice_arbol: int = 0,
) -> dict:
    paciente = preparar_paciente(datos)
    prediccion = str(modelo.predict(paciente)[0])
    probabilidades = modelo.predict_proba(paciente)[0]
    clase = _normalizar_clase(prediccion)

    return {
        "prediccion": prediccion,
        "clase": clase,
        "probabilidades": {
            _normalizar_clase(clase_modelo): float(probabilidad)
            for clase_modelo, probabilidad in zip(modelo.classes_, probabilidades)
        },
        "explicacion_arbol": explicar_prediccion(
            modelo,
            paciente,
            indice_arbol=indice_arbol,
        ),
    }


def explicar_prediccion(
    modelo,
    datos_paciente: pd.DataFrame,
    indice_arbol: int = 0
):
    if not 0 <= indice_arbol < len(modelo.estimators_):
        raise ValueError("Índice de árbol fuera de rango.")

    if list(datos_paciente.columns) != VARIABLES_PREDICTORAS:
        raise ValueError(
            "Las columnas del paciente no coinciden con "
            "las variables predictoras del modelo."
        )

    arbol = modelo.estimators_[indice_arbol]
    estructura = arbol.tree_

    datos_arbol = datos_paciente.to_numpy()
    nodos_visitados = arbol.decision_path(datos_arbol).indices.tolist()

    nodo_hoja = int(arbol.apply(datos_arbol)[0])

    visitados = set(nodos_visitados)
    nodos = []

    for nodo_id in range(estructura.node_count):
        feature_id = estructura.feature[nodo_id]
        es_hoja = feature_id < 0

        if not es_hoja:
            variable = VARIABLES_PREDICTORAS[feature_id]
            umbral = float(estructura.threshold[nodo_id])
            valor = float(datos_paciente.iloc[0, feature_id])
            nombre_variable = (
                f"log({variable})" if variable in VARIABLES_LOG else variable
            )
            condicion = f"{nombre_variable} <= {umbral:.4f}"
            if nodo_id in visitados:
                rama = "izquierda" if valor <= umbral else "derecha"
                evaluacion = (
                    f"{nombre_variable}={valor:.4f}; "
                    f"{condicion}; rama {rama}"
                )
            else:
                rama = None
                evaluacion = None
        else:
            variable = None
            nombre_variable = None
            umbral = None
            valor = None
            condicion = "Clasificación final"
            rama = None
            evaluacion = "Hoja alcanzada" if nodo_id == nodo_hoja else None

        valores = estructura.value[nodo_id][0].tolist()
        clase_nodo = _normalizar_clase(modelo.classes_[int(np.argmax(valores))])

        nodos.append({
            "id": int(nodo_id),
            "variable": nombre_variable,
            "umbral": umbral,
            "valor_paciente": valor,
            "condicion": condicion,
            "condicion_evaluada": evaluacion,
            "izquierda": (
                int(estructura.children_left[nodo_id])
                if not es_hoja else None
            ),
            "derecha": (
                int(estructura.children_right[nodo_id])
                if not es_hoja else None
            ),
            "muestras": int(estructura.n_node_samples[nodo_id]),
            "impureza": float(estructura.impurity[nodo_id]),
            "clase": clase_nodo,
            "es_hoja": bool(es_hoja),
            "visitado": nodo_id in visitados,
            "rama_seguida": rama,
        })

    return {
        "indice_arbol": int(indice_arbol),
        "nodos": nodos,
        "nodos_visitados": nodos_visitados,
        "nodo_hoja": nodo_hoja,
        "total_nodos": estructura.node_count,
        "profundidad": int(estructura.max_depth),
        "total_arboles": len(modelo.estimators_),
    }

if __name__ == "__main__":
    print("=== ENTRENANDO MODELO DE CLASIFICACIÓN (RANDOM FOREST) ===")
    res = entrenar_random_forest()
    
    print(f"\nExactitud (Accuracy): {res['accuracy']:.4f}\n")
    print("Matriz de Confusión:")
    print(res["confusion_matrix"])
    print("\nReporte de Clasificación:")
    print(res["classification_report"])
    print("\nImportancia de las Variables:")
    print(res["feature_importances"])