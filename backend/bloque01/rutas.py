from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from .clasificacion import VARIABLES_LOG, VARIABLES_PREDICTORAS, entrenar_random_forest
from .datos import estadisticas_datos, resumen_datos


router = APIRouter(prefix="/api/b1", tags=["bloque01"])


class DatosPaciente(BaseModel):
    edad: float = Field(alias="Age", gt=0, le=120)
    frecuencia_cardiaca: float = Field(alias="Heart rate", gt=0, le=300)
    presion_sistolica: float = Field(alias="Systolic blood pressure", gt=0, le=350)
    presion_diastolica: float = Field(alias="Diastolic blood pressure", gt=0, le=250)
    glucosa: float = Field(alias="Blood sugar", gt=0, le=2000)
    ck_mb: float = Field(alias="CK-MB", ge=0)
    troponina: float = Field(alias="Troponin", ge=0)


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
def predecir_paciente(paciente: DatosPaciente):
    datos = paciente.dict(by_alias=True)
    entrada = pd.DataFrame([datos], columns=VARIABLES_PREDICTORAS)
    for variable in VARIABLES_LOG:
        entrada[variable] = np.log(entrada[variable] + 1e-5)

    modelo = _obtener_modelo()
    prediccion = str(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    indice_prediccion = list(modelo.classes_).index(prediccion)

    return {
        "prediction": prediccion,
        "probability": float(probabilidades[indice_prediccion]),
    }