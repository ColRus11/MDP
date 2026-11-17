# Análisis cardiovascular: MDP y CardioLab

Este repositorio reúne dos desarrollos académicos basados en el conjunto
`Medicaldataset.csv`: un modelo de decisión de Markov para estudiar estrategias
de triaje y una aplicación de análisis que combina métodos de aprendizaje
automático con una interfaz web.

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
- entrenar un clasificador Random Forest y obtener predicciones experimentales.

La carpeta `frontend/` contiene una interfaz estática en HTML, CSS y JavaScript.
Consulta los endpoints disponibles para mostrar el resumen y las estadísticas,
y permite enviar datos a la ruta de predicción.

En esta versión, las vistas de clustering y del resumen del clasificador también
solicitan rutas que aún no están implementadas en la API
(`/api/b1/clustering/jerarquico`, `/api/b1/clustering/perfiles` y
`/api/b1/clasificacion/resumen`). Esos paneles pueden permanecer pendientes.

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
- Joblib para los scripts de modelado existentes en la raíz.

El frontend no requiere paquetes de Node.js ni un proceso de compilación; utiliza
HTML, CSS y JavaScript nativos.

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
| `POST` | `/api/b1/clasificacion/predecir` | Predicción experimental para un registro |

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

## Estructura principal

```text
.
├── backend/
│   ├── bloque01/       # Datos, clustering, clasificación y rutas de la API
│   ├── data/           # Conjunto de datos utilizado por CardioLab
│   ├── figuras/        # Dendrograma de referencia
│   └── main.py         # Aplicación FastAPI
├── frontend/           # Interfaz web estática
├── taller3_dqn/        # Extensión de aprendizaje por refuerzo profundo
├── estados.py          # Preparación de estados para el MDP
├── recompensa.py       # Matriz de recompensa
├── transiciones.py     # Estimación de transiciones
└── politica_mdp.py     # Iteración de valor y política
```

## Alcance

Los modelos se entrenan con el conjunto disponible y tienen propósito
demostrativo. La calidad de sus predicciones depende de la representatividad,
procedencia y limitaciones de esos datos. No deben utilizarse para decidir la
atención de pacientes ni sustituir la evaluación de profesionales de la salud.
