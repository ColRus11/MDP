# CardioLab frontend

Interfaz estática en HTML, CSS y JavaScript. No entrena modelos ni calcula métricas: solicita los datos al backend y solo los presenta.

## Inicio

Sirve la carpeta raíz del proyecto para que la figura relativa del dendrograma pueda cargarse:

```bash
python3 -m http.server 5500
```

Abre `http://localhost:5500/frontend/`. La API se configura en `frontend/app.js` con `API_BASE` (por defecto `http://127.0.0.1:8000`). Si el frontend y FastAPI usan orígenes distintos, el backend debe permitir el origen del frontend mediante CORS o un proxy de desarrollo.

## Endpoints utilizados

- `GET /api/b1/datos/resumen`: filas, columnas, valores nulos, duplicados y distribución de la etiqueta.
- `GET /api/b1/datos/estadisticas`: estadísticos descriptivos de las variables.
- `POST /api/b1/clustering/asignar`: asignación K-Means `k=5` del formulario de siete variables.
- `GET /api/b1/clasificacion/resumen`: métricas del conjunto de prueba, matriz de confusión, reporte e importancia de variables.
- `POST /api/b1/clasificacion/predecir`: predicción, probabilidades estimadas por clase y estructura/recorrido del árbol seleccionado.
- `GET /api/b1/dbscan/analisis?eps=2&min_samples=14`: conteos, perfil clínico y comparaciones para DBSCAN. Los controles de la página permiten cambiar ambos parámetros.

Las rutas opcionales para métricas jerárquicas y perfiles tabulares de clustering
(`GET /api/b1/clustering/jerarquico` y `GET /api/b1/clustering/perfiles`) aún no
están implementadas; sus paneles conservan el estado pendiente.

El Random Forest se entrena una sola vez por proceso. Las predicciones, las
métricas y el árbol individual mostrado utilizan ese mismo estimador. La
visualización del árbol es interactiva: permite seleccionar nodos, hacer zoom
y desplazar el área de trabajo.

FastAPI permite CORS para los orígenes locales `localhost` y `127.0.0.1`. Sirve el frontend por HTTP (no abras el HTML directamente como `file://`) para que el navegador envíe un origen local permitido.

El navegador usa la imagen `../backend/figuras/dendrograma_ward.png`; el servidor estático debe exponer la carpeta `backend/` desde la raíz del proyecto.
