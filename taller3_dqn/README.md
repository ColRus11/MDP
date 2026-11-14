# Taller 3 · Parte 3 — Prototipo: del Q-Learning tabular a una Deep Q-Network

Evolución del agente de triaje de infarto del **Taller 2** (Q-Learning con Q-Table) a una **DQN** con
Experience Replay y Red Objetivo, con una aplicación que lo muestra funcionando como en un servicio de urgencias.

Se conserva el problema del Taller 2 (mismo entorno, misma matriz de recompensa, mismos hiperparámetros: γ = 0.9,
ε con decaimiento 0.999, 10 000 pasos) y la limpieza del Taller 1 (1 314 pacientes). Solo cambia el "cerebro" del agente.

| Resultado real | Verde | Amarillo | Rojo |
|---|---|---|---|
| Infarto | −50 | +2 | +10 |
| Sano | +5 | +0.5 | −3 |

## Estructura

| Archivo | Qué es |
|---|---|
| `app.py` | **Aplicación** (Streamlit): Urgencias en vivo, Evaluar paciente, Entrenamiento en vivo y Resultados |
| `triaje_rl/core.py` | Núcleo: datos, entorno, agente Q-Table (Taller 2) y agente DQN (Taller 3) |
| `entrenar.py` | Entrena los agentes, corre los experimentos y guarda `modelos/` y `resultados/resultados_app.json` |
| `Taller3_DQN_PersonaB.ipynb` | El laboratorio paso a paso (funciona en Google Colab: descarga el dataset solo) |
| `modelos/` | Agentes ya entrenados (la app los carga) |
| `resultados/` | Gráficas y tablas del experimento |

## Cómo ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                  # abre http://localhost:8501
```

`python entrenar.py` vuelve a entrenar todo (≈ 1 minuto). Si falta la carpeta `modelos/`, la app entrena sola al abrir.

## La aplicación

1. **🚑 Urgencias en vivo** — llegan pacientes que el agente nunca vio; la DQN decide el semáforo en tiempo real y la
   Q-Table del Taller 2 decide en paralelo sobre los mismos pacientes. Contadores de falsas alarmas, infartos enviados a
   Verde, recompensa acumulada y alertas de casos críticos.
2. **🩺 Evaluar paciente** — el flujo del mockup: datos → decisión con los valores Q de cada color y la ubicación del
   paciente en el mapa de decisiones → validación e intervención humana (override con razón, guardado en `registros/`).
3. **🧠 Entrenamiento en vivo** — entrena una DQN desde cero viendo cómo aprende; se pueden apagar Experience Replay y la
   Red Objetivo o cambiar las variables. El agente queda disponible en el selector de la barra lateral.
4. **📊 Resultados del experimento** — comparación de políticas, mapas de decisión y los dos experimentos.

## Resultados (263 pacientes de prueba, nunca vistos)

| | Recompensa media | Falsas alarmas | Infartos en Verde | Sanos en Verde |
|---|---|---|---|---|
| Q-Table (Taller 2) | 5.33 | 38 | 0 | 0 % |
| DQN · 8 variables | 6.73 | 27 | 1 | 59 % |
| **DQN final · troponina + CK-MB** | **7.88** | **5** | **0** | **92 %** |

| Experimento de estabilidad (3 semillas, 8 variables) | Recompensa media | Infartos en Verde (prom. / máx.) |
|---|---|---|
| DQN completa | 6.67 ± 0.24 | 0.3 / 1 |
| Sin Experience Replay | 5.48 ± 1.02 | 3.0 / 7 |
| Sin Red Objetivo | 6.67 ± 0.30 | 1.0 / 1 |

**Conclusiones:** la mejora viene de ver los marcadores **continuos** (no de ver más variables: con ~1 000 pacientes las
6 variables extra solo agregan ruido); Experience Replay es indispensable; la Red Objetivo pesa poco aquí porque el
siguiente paciente no depende de la acción. El caso ambiguo del mockup (troponina 0.005, CK-MB 0.676, con infarto) sale
en Verde: por eso el sistema es apoyo a la decisión y requiere la intervención humana.
