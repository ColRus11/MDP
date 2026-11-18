from pathlib import Path
import pandas as pd
import numpy as np

# Importamos las funciones de tus módulos existentes
from bloque01.datos import preparar_clustering, limpiar_datos
from bloque01.clustering import evaluar_jerarquico, perfilar_kmeans, perfilar_jerarquico, guardar_dendrograma
from bloque01.clasificacion import entrenar_random_forest

def ejecutar_pipeline_completo():
    print("=" * 60)
    print(" INICIANDO PIPELINE COMPLETO: BLOQUE 01 (CLUSTERING + CLASIFICACIÓN)")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. PARTE NO SUPERVISADA: CLUSTERING Y DENDROGRAMA
    # -------------------------------------------------------------
    print("\n[1/4] Preparando datos para clustering...")
    _, X_clus, _ = preparar_clustering()
    
    print("[2/4] Evaluando Clustering Jerárquico y guardando dendrograma...")
    resultados_hier = evaluar_jerarquico(X_clus)
    
    # Guardar la imagen del dendrograma
    ruta_fig = Path("backend/figuras/dendrograma_ward.png")
    guardar_dendrograma(X_clus, metodo="ward", ruta=str(ruta_fig))
    print(f" -> Dendrograma guardado exitosamente en: {ruta_fig}")
    
    # Perfil jerárquico de ejemplo (Ward, k=3)
    print("\n--- Perfil Clínico: Jerárquico (Ward, k=3) ---")
    perfil_jer = perfilar_jerarquico(k=3, metodo="ward")
    print(perfil_jer)
    
    # Perfil K-Means de ejemplo (k=5)
    print("\n--- Perfil Clínico: K-Means (k=5) ---")
    perfil_km = perfilar_kmeans(k=5)
    print(perfil_km)
    
    # -------------------------------------------------------------
    # 2. PARTE SUPERVISADA: CLASIFICACIÓN (RANDOM FOREST)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3/4] Entrenando y evaluando modelo de Clasificación (Random Forest)...")
    print("=" * 60)
    
    res_rf = entrenar_random_forest()
    
    print(f"\nExactitud (Accuracy) del Modelo: {res_rf['accuracy']:.4%}")
    
    print("\nMatriz de Confusión:")
    print(res_rf["confusion_matrix"])
    
    print("\nReporte de Clasificación:")
    print(res_rf["classification_report"])
    
    print("\nImportancia de las Variables:")
    print(res_rf["feature_importances"])
    
    print("\n" + "=" * 60)
    print(" ¡PIPELINE EJECUTADO CON ÉXITO PARA LA DEMO!")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_pipeline_completo()