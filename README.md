# Análisis cardiovascular: MDP y CardioLab

Este repositorio reúne tres desarrollos académicos basados en el conjunto
`Medicaldataset.csv`: un modelo de decisión de Markov para estudiar estrategias
de triaje, CardioLab para explorar datos y modelos de aprendizaje automático,
y un prototipo de triaje con aprendizaje por refuerzo profundo.

El trabajo tiene fines educativos y de exploración metodológica. Sus salidas no
constituyen diagnósticos ni recomendaciones clínicas.

## Componentes

### Modelo de decisión de Markov

Los scripts de la raíz construyen estados a partir de marcadores cardíacos,
estiman recompensas y transiciones, y calculan una política mediante iteración de
valor. El modelo representa cuatro estados de riesgo y tres acciones:
`ESCALAR`, `REPETIR` y `SEGUIMIENTO`. Las transiciones se estiman mediante una
simulación descrita en `transiciones.py`; no representan observaciones
longitudinales de pacientes.

### CardioLab

La carpeta `backend/` contiene una API en FastAPI y módulos para:

- revisar la estructura, calidad y estadísticas descriptivas del conjunto;
- preparar variables clínicas con transformaciones logarítmicas y
  estandarización;
- explorar agrupamientos K-Means y jerárquicos mediante perfiles clínicos,
  silhouette, índice de Dunn y correlación cofenética;
- entrenar un clasificador Random Forest y obtener predicciones experimentales;
- detectar observaciones atípicas con DBSCAN y comparar sus perfiles con el
  resto del conjunto.

La carpeta `frontend/` contiene una interfaz estática en HTML, CSS y JavaScript.
Consulta la API para presentar el resumen y las estadísticas de los datos, las
métricas del clasificador, las probabilidades estimadas por clase, los análisis
DBSCAN y la asignación del paciente a un grupo K-Means. La explicación
individual incluye el recorrido del paciente en un árbol del bosque. La
transformación de datos, el entrenamiento y los cálculos se realizan en Python;
el navegador se limita a enviar entradas y presentar respuestas.

### Taller 3: aprendizaje por refuerzo profundo

`taller3_dqn/` contiene una aplicación independiente de triaje que compara un
agente Q-Learning tabular con una Deep Q-Network (DQN). La red utiliza
Experience Replay y una red objetivo; el subproyecto también incluye el código
de entrenamiento, experimentos, resultados y un cuaderno de trabajo. Sus
decisiones son demostrativas y no sustituyen la evaluación clínica.

## Datos

El archivo de entrada contiene variables demográficas y clínicas —edad, género,
frecuencia cardíaca, presión arterial, glucosa, CK-MB y troponina— junto con la
etiqueta `Result`. CardioLab carga la copia ubicada en
`backend/data/Medicaldataset.csv`; los scripts de MDP usan `Medicaldataset.csv`
en la raíz.

Antes de redistribuir el conjunto fuera de este repositorio, verifique las
condiciones de uso y citación establecidas por su fuente original.

## Requisitos

Se requiere Python 3.9 o posterior. Las dependencias de Python están declaradas
en [`requirements.txt`](requirements.txt):

- FastAPI y Uvicorn para la API;
- Pydantic para validar las entradas;
- NumPy, pandas y SciPy para manipulación de datos y cálculos numéricos;
- scikit-learn para clasificación, agrupamiento y métricas;
- Matplotlib para generar figuras;
- Joblib para guardar y cargar modelos.

El frontend no requiere paquetes de Node.js ni un proceso de compilación; utiliza
HTML, CSS y JavaScript nativos.

El proyecto DQN mantiene dependencias separadas en
[`taller3_dqn/requirements.txt`](taller3_dqn/requirements.txt), entre ellas
PyTorch, Streamlit, Altair y Jupyter. No es necesario instalarlas para ejecutar
el MDP de la raíz o CardioLab.

### Instalación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell, active el entorno con:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución del taller DQN

Para instalar y ejecutar la aplicación DQN, use un entorno virtual independiente:

```bash
cd taller3_dqn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`. Para volver a ejecutar los
experimentos y generar sus artefactos, ejecute `python entrenar.py`. El cuaderno
`Taller3_DQN.ipynb` presenta el desarrollo paso a paso.

## Ejecución de CardioLab

Desde la raíz, inicie la API:

```bash
uvicorn main:app --app-dir backend --reload
```

La documentación interactiva de FastAPI queda disponible en
`http://127.0.0.1:8000/docs`. Rutas implementadas:

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/api/salud` | Comprobación del servicio |
| `GET` | `/api/b1/datos/resumen` | Dimensiones, nulos, duplicados y distribución de la etiqueta |
| `GET` | `/api/b1/datos/estadisticas` | Estadísticos descriptivos de las variables |
| `GET` | `/api/b1/clasificacion/resumen` | Métricas sobre el conjunto de prueba e importancia de variables |
| `POST` | `/api/b1/clasificacion/predecir` | Predicción, estimaciones por clase y recorrido en un árbol del Random Forest |
| `POST` | `/api/b1/clustering/asignar` | Asignación K-Means y perfil del grupo de entrenamiento |
| `GET` | `/api/b1/dbscan/analisis` | Perfil comparativo de ruido y resto (`eps` y `min_samples` configurables) |
| `GET` | `/api/b1/clustering/jerarquico` | Silueta, correlación cofenética y tamaño mínimo por enlace y `k` |
| `GET` | `/api/b1/clustering/perfiles` | Perfil clínico por grupo (`metodo=kmeans` o enlace jerárquico, `k`) |
| `GET` | `/api/b2/regresion/resumen` | Regresión del riesgo de infarto: R², MAE, RMSE, importancias y puente al MDP |
| `GET` | `/api/b2/mdp/resumen` | Estados, recompensa, transiciones, política óptima y batalla de políticas |
| `GET` | `/api/b2/rl/resumen` | Q-Learning tabular frente a DQN: comparación y experimentos de `taller3_dqn` |

### Pestañas de la aplicación

La interfaz reúne todo el proyecto en un solo programa:

| Pestaña | Contenido |
|---|---|
| Introducción | Problema, recorrido del pipeline y equipo |
| Resumen y datos | Calidad y estadísticas del conjunto |
| Clustering | K-Means, jerárquico y DBSCAN |
| Clasificación · Demo predictiva | Random Forest y consulta de un paciente |
| Regresión | Riesgo continuo de infarto y su conexión con el MDP |
| MDP · Iteración de valor | Estados, R, P, política óptima y batalla |
| Q-Learning y DQN | Comparación de agentes y experimentos de estabilidad |
| Conclusiones | Hallazgos, limitaciones y trabajo futuro |

Las pestañas MDP y Q-Learning/DQN leen los artefactos ya generados en la raíz y en
`taller3_dqn/resultados/`; si faltan, vuelva a ejecutar los scripts correspondientes.

El Random Forest se entrena una sola vez por proceso y se reutiliza para la
predicción y el reporte de evaluación. La visualización representa un árbol del
bosque, no el modelo completo; la predicción final combina sus 100 árboles.
El modelo y los resultados de evaluación se guardan en
`backend/models/random_forest.joblib`; el artefacto se invalida si cambian la
configuración, la versión de scikit-learn o el hash del conjunto de datos.
Las predicciones y los grupos DBSCAN son resultados experimentales, no
diagnósticos clínicos.

En otra terminal, sirva la carpeta del proyecto para abrir la interfaz:

```bash
python3 -m http.server 5500
```

Visite `http://localhost:5500/frontend/`. La interfaz utiliza por defecto la API
en `http://127.0.0.1:8000`; ambos servicios deben permanecer activos.

## Reproducción de los análisis

Para ejecutar el pipeline de CardioLab desde la raíz:

```bash
PYTHONPATH=backend python -m bloque01.ejecutar
```

En Windows PowerShell:

```powershell
$env:PYTHONPATH = "backend"
python -m bloque01.ejecutar
```

La regresión del Bloque 2 se reproduce así (genera `backend/bloque02/resultados/`):

```bash
cd backend
python -m bloque02.regresion
```

Los scripts del modelo MDP también pueden ejecutarse desde la raíz, siguiendo
este orden:

```bash
python estados.py
python recompensa.py
python transiciones.py
python politica_mdp.py
python grafica_politica.py
```

El subproyecto `taller3_dqn/` conserva su propio archivo de dependencias y sus
instrucciones de ejecución en su README.

Las pruebas focalizadas del endpoint de asignación pueden ejecutarse así:

```bash
cd backend
python3 -m unittest discover -s tests -v
```

## Estructura principal

```text
.
├── backend/
│   ├── bloque01/       # Datos, clustering, clasificación, esquemas y rutas
│   ├── bloque02/       # Regresión de riesgo y rutas de MDP y refuerzo
│   ├── chequeo.py      # Tabla cruzada de una partición jerárquica
│   ├── data/           # Conjunto de datos utilizado por CardioLab
│   ├── figuras/        # Dendrograma y gráfica de k-distancias
│   ├── tests/          # Pruebas de clustering y análisis de modelos
│   └── main.py         # Aplicación FastAPI
├── frontend/           # Interfaz web estática (app.js, styles.css, ensamblado.css)
├── taller3_dqn/        # Extensión de aprendizaje por refuerzo profundo
├── estados.py          # Preparación de estados para el MDP
├── recompensa.py       # Matriz de recompensa
├── transiciones.py     # Estimación de transiciones
├── politica_mdp.py     # Iteración de valor y política
└── grafica_politica.py # Visualización de la política
```

## Alcance

Los modelos se entrenan con el conjunto disponible y tienen propósito
demostrativo. La calidad de sus predicciones depende de la representatividad,
procedencia y limitaciones de esos datos. No deben utilizarse para decidir la
atención de pacientes ni sustituir la evaluación de profesionales de la salud.
