"""Rutas del Bloque 2 y del aprendizaje por refuerzo (ensamblado de CardioLab).

- /api/b2/regresion/resumen : regresión del índice de riesgo de infarto.
- /api/b2/mdp/resumen       : MDP + Iteración de Valor (estados, R, P, política, batalla).
- /api/b2/rl/resumen        : Q-Learning tabular vs DQN (resultados de taller3_dqn).

Las rutas MDP y RL leen los artefactos ya generados en el repositorio, por lo que no
reentrenan nada al consultarse.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from .regresion import entrenar_regresion, mapa_riesgo_accion

router = APIRouter(prefix="/api/b2", tags=["bloque02"])

RAIZ = Path(__file__).resolve().parents[2]          # raíz del repositorio
DIR_RL = RAIZ / "taller3_dqn" / "resultados"
ACCIONES = ["ESCALAR", "REPETIR", "SEGUIMIENTO"]


def _leer_csv(ruta: Path, **kwargs) -> pd.DataFrame:
    if not ruta.exists():
        raise HTTPException(status_code=503, detail=f"Falta el archivo {ruta.name}. Ejecute los scripts del MDP/RL.")
    return pd.read_csv(ruta, **kwargs)


@router.get("/regresion/resumen")
def regresion_resumen():
    datos = entrenar_regresion()
    return {
        "objetivo": "Riesgo continuo de infarto (0 = sano, 1 = infarto)",
        "resultados": datos["resultados"],
        "mejor_modelo": datos["mejor_modelo"],
        "importancias": datos["importancias"],
        "n_test": datos["n_test"],
        "puente_mdp": [
            {"riesgo": r, "accion": mapa_riesgo_accion(r)} for r in (0.1, 0.5, 0.8)
        ],
    }


@lru_cache(maxsize=1)
def _mdp() -> dict:
    perfil = _leer_csv(RAIZ / "perfil_estados.csv")
    recompensa = _leer_csv(RAIZ / "recompensa.csv", index_col=0)
    politica = _leer_csv(RAIZ / "politica_optima.csv")
    batalla = _leer_csv(RAIZ / "batalla_politicas.csv")
    transiciones = {
        a: _leer_csv(RAIZ / f"P_{a}.csv", index_col=0) for a in ACCIONES
    }
    return {
        "estados": [
            {
                "estado": int(f["estado"]),
                "nombre": str(f["estado_nombre"]),
                "pacientes": int(f["n"]),
                "troponina_mediana": float(f["troponina_mediana"]),
                "ckmb_mediana": float(f["ckmb_mediana"]),
                "pct_infarto": float(f["pct_positivo"]),
            }
            for _, f in perfil.iterrows()
        ],
        "acciones": ACCIONES,
        "recompensa": [
            {"estado": str(est), **{a: float(recompensa.loc[est, a]) for a in ACCIONES}}
            for est in recompensa.index
        ],
        "politica": [
            {
                "estado": str(f["estado"]),
                "v_optimo": float(f["V_optimo"]),
                "accion_optima": str(f["accion_optima"]),
                **{f"q_{a}": float(f[f"Q_{a}"]) for a in ACCIONES},
            }
            for _, f in politica.iterrows()
        ],
        "batalla": [
            {"politica": str(f["politica"]), "v_promedio": float(f["V_promedio"])}
            for _, f in batalla.iterrows()
        ],
        "transiciones": {
            a: {
                "estados": [str(c) for c in tabla.columns],
                "matriz": tabla.round(2).values.tolist(),
            }
            for a, tabla in transiciones.items()
        },
        "gamma": 0.9,
    }


@router.get("/mdp/resumen")
def mdp_resumen():
    return _mdp()


@lru_cache(maxsize=1)
def _rl() -> dict:
    comparacion = _leer_csv(DIR_RL / "5_comparacion_final.csv", index_col=0)
    estabilidad = _leer_csv(DIR_RL / "experimento_estabilidad.csv")
    variables = _leer_csv(DIR_RL / "experimento_variables.csv")
    app = json.loads((DIR_RL / "resultados_app.json").read_text(encoding="utf-8"))
    return {
        "n_entrenamiento": int(app.get("n_entrenamiento", 0)),
        "n_prueba": int(app.get("n_prueba", 0)),
        "comparacion": [
            {
                "politica": str(nombre),
                "recompensa": float(f["Recompensa clínica media"]),
                "falsos_negativos": int(f["Falsos negativos (infarto en Verde)"]),
                "falsas_alarmas": int(f["Falsas alarmas (sano en Rojo)"]),
                "sanos_en_verde": str(f["Sanos en Verde"]),
                "pct_rojo": str(f["% Rojo"]),
                "pct_amarillo": str(f["% Amarillo"]),
                "pct_verde": str(f["% Verde"]),
            }
            for nombre, f in comparacion.iterrows()
        ],
        "estabilidad": estabilidad.to_dict(orient="records"),
        "variables": variables.to_dict(orient="records"),
        "recompensa_triaje": {
            "Infarto": {"Verde": -50, "Amarillo": 2, "Rojo": 10},
            "Sano": {"Verde": 5, "Amarillo": 0.5, "Rojo": -3},
        },
    }


@router.get("/rl/resumen")
def rl_resumen():
    return _rl()
