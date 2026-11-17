from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .datos import limpiar_datos  # Reutiliza tu función de limpieza existente

SEMILLA = 42

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

def entrenar_random_forest():
    X, y = preparar_datos_clasificacion()
    
    # 1. División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y
    )
    
    # 2. Instanciar y entrenar Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
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
    
    # 5. Importancia de las variables
    importancias = pd.Series(rf.feature_importances_, index=VARIABLES_PREDICTORAS).sort_values(ascending=False)
    
    resultados = {
        "accuracy": acc,
        "confusion_matrix": matriz_conf,
        "classification_report": reporte,
        "feature_importances": importancias,
        "modelo": rf
    }
    
    return resultados

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