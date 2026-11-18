from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "Medicaldataset.csv"

RELLENO = [
    (1111,141,95),
    (20,110,90),]


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_CSV)

def limpiar_datos()-> pd.DataFrame:
    df = cargar_datos()
    es_relleno = pd.Series(False, index=df.index)
    for pulso, sistolica, diastolica in RELLENO:
        es_relleno |= (
            (df["Heart rate"] == pulso)
            & (df["Systolic blood pressure"] == sistolica)
            & (df["Diastolic blood pressure"] == diastolica)
        )
    return df[~es_relleno].reset_index(drop=True)


def resumen_datos() -> dict:
    df = limpiar_datos()
    objetivo = df.columns[-1]  # supuesto: la última columna; verifícalo
    return {
        "filas": int(df.shape[0]),
        "columnas": list(df.columns),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "nulos": {c: int(n) for c, n in df.isna().sum().items()},
        "duplicados": int(df.duplicated().sum()),
        "objetivo": objetivo,
        "distribucion_objetivo": {
            str(k): int(v) for k, v in df[objetivo].value_counts().items()
        },
    }

def estadisticas_datos() -> dict:
    df = limpiar_datos()
    num = df.drop(columns=["Result"])
    desc = num.describe().T
    return {
        c: {
            "min": float(desc.loc[c, "min"]),
            "mediana": float(desc.loc[c,"50%"]),
            "media": float(desc.loc[c,"mean"]),
            "max": float(desc.loc[c, "max"]),
            "asimetria": float(num[c].skew()),
        }
        for c in num.columns
    }
VARIABLES_CLUSTERING = [
    "Age", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Blood sugar", "CK-MB", "Troponin",
]
VARIABLES_LOG = ["CK-MB", "Troponin"]

def preparar_clustering():
    df = limpiar_datos()
    X = df[VARIABLES_CLUSTERING].copy()
    for c in VARIABLES_LOG:
        X[c] = np.log(X[c])
    escalador = StandardScaler()
    X_esc = escalador.fit_transform(X)
    return df, X_esc, escalador

if __name__ == "__main__":
    df, X_esc, _ = preparar_clustering()
    print("Forma:", X_esc.shape)
    print("Medias (≈0):", X_esc.mean(axis=0).round(3))
    print("Desviaciones (≈1):", X_esc.std(axis=0).round(3))
    print("Asimetría tras log:\n", np.log(df[VARIABLES_LOG]).skew())
