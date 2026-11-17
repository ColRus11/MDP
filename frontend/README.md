# CardioLab frontend

Interfaz estática en HTML, CSS y JavaScript. No entrena modelos ni calcula métricas: solicita los datos al backend y solo los presenta.

## Inicio

Sirve la carpeta raíz del proyecto para que la figura relativa del dendrograma pueda cargarse:

```bash
python3 -m http.server 5500
```

Abre `http://localhost:5500/frontend/`. La API se configura en `frontend/app.js` con `API_BASE` (por defecto `http://127.0.0.1:8000`). Si el frontend y FastAPI usan orígenes distintos, el backend debe permitir el origen del frontend mediante CORS o un proxy de desarrollo.

## Endpoints utilizados

Ya disponibles en el backend:

- `GET /api/b1/datos/resumen`: `filas`, `columnas`, `nulos`, `duplicados`, `objetivo` y `distribucion_objetivo`.
- `GET /api/b1/datos/estadisticas`: objeto indexado por variable con `min`, `mediana`, `media`, `max` y `asimetria`.

El backend actual todavía no implementa las rutas siguientes; sus paneles muestran estados pendientes hasta que estén disponibles:

- `GET /api/b1/clustering/jerarquico`: lista de `{ metodo, k, silhouette, cofenetica, tamano_min }`.
- `GET /api/b1/clustering/perfiles?metodo=kmeans&k=5`: lista o `{ perfiles: [...] }` con `cluster`, `pacientes`, `edad`, `pulso`, `sistolica`, `glucosa`, `ckmb_mediana`, `troponina_mediana` y `pct_infarto`.
- `GET /api/b1/clasificacion/resumen`: `{ accuracy, confusion_matrix, clases, classification_report, feature_importances }`. `accuracy` se expresa entre 0 y 1.
- `POST /api/b1/clasificacion/predecir` está implementado: recibe las variables predictoras del formulario (`Age`, `Heart rate`, `Systolic blood pressure`, `Diastolic blood pressure`, `Blood sugar`, `CK-MB`, `Troponin`) y responde `{ prediction, probability }`. `probability` es la confianza del modelo para la clase predicha, entre 0 y 1. La primera llamada entrena y conserva el modelo en memoria.

FastAPI permite CORS para los orígenes locales `localhost` y `127.0.0.1`. Sirve el frontend por HTTP (no abras el HTML directamente como `file://`) para que el navegador envíe un origen local permitido.

El navegador usa la imagen `../backend/figuras/dendrograma_ward.png`; el servidor estático debe exponer la carpeta `backend/` desde la raíz del proyecto.
