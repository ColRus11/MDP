"""Entrena los agentes, corre los experimentos y guarda todo lo que usa la app.

Uso:  python entrenar.py
"""
import json
import time

import numpy as np
import pandas as pd

from triaje_rl.core import (CARPETA_MODELOS, CARPETA_RESULTADOS, FEATURES, MARCADORES, AgenteDQN,
                            AgenteQTable, cargar_datos, evaluar, media_movil)

SEMILLAS = [0, 1, 2]


def curva(v, n=300, puntos=200):
    """Media móvil reducida a `puntos` valores para graficar en la app."""
    m = media_movil(v, n)[n:]
    idx = np.linspace(0, len(m) - 1, puntos).astype(int)
    return {'x': (idx + n).tolist(), 'y': np.round(m[idx], 4).tolist()}


def experimento(tr, te, variantes):
    filas, curvas = [], {}
    for nombre, cfg in variantes.items():
        columnas = cfg.get('columnas', FEATURES)
        opciones = {k: v for k, v in cfg.items() if k != 'columnas'}
        evals, rec, perd, qm = [], [], [], []
        for s in SEMILLAS:
            agente = AgenteDQN(columnas)
            hist = agente.entrenar(tr, seed=s, **opciones)
            evals.append(evaluar(agente.decidir(te), te))
            rec.append(curva(hist['recompensas'])); perd.append(curva(hist['perdidas'], 200))
            qm.append(curva(hist['q_medio'], 200))
        e = pd.DataFrame(evals)
        filas.append({'Variante': nombre,
                      'Recompensa media': e['Recompensa clínica media'].mean(),
                      'Recompensa desv.': e['Recompensa clínica media'].std(),
                      'Infartos en Verde (prom.)': e['Infartos en Verde'].mean(),
                      'Infartos en Verde (máx.)': int(e['Infartos en Verde'].max()),
                      'Falsas alarmas (prom.)': e['Falsas alarmas'].mean()})
        curvas[nombre] = {'recompensa': rec, 'perdida': perd, 'q_medio': qm}
        print(f'  {nombre}: {filas[-1]["Recompensa media"]:.2f} · infartos en Verde máx {filas[-1]["Infartos en Verde (máx.)"]}')
    return filas, curvas


def main():
    t0 = time.time()
    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    df, tr, te = cargar_datos()
    print(f'{len(tr)} pacientes de entrenamiento · {len(te)} de prueba')

    qtable = AgenteQTable(tr)
    hist_q = qtable.entrenar(tr)
    qtable.guardar(CARPETA_MODELOS / 'qtable_taller2.npz')

    dqn8 = AgenteDQN(FEATURES)
    hist_8 = dqn8.entrenar(tr)
    dqn8.guardar(CARPETA_MODELOS / 'dqn_8_variables.pt')

    dqn = AgenteDQN(MARCADORES)
    hist_2 = dqn.entrenar(tr)
    dqn.guardar(CARPETA_MODELOS / 'dqn_final.pt')

    comparacion = {
        'Siempre Rojo': evaluar(np.full(len(te), 2), te),
        qtable.nombre: evaluar(qtable.decidir(te), te),
        dqn8.nombre: evaluar(dqn8.decidir(te), te),
        'DQN final · troponina + CK-MB': evaluar(dqn.decidir(te), te),
    }
    for k, v in comparacion.items():
        print(f'  {k:32s} recompensa {v["Recompensa clínica media"]:.2f} · falsas alarmas {v["Falsas alarmas"]:3d}'
              f' · infartos en Verde {v["Infartos en Verde"]}')

    print('Experimento de estabilidad (8 variables, 3 semillas)...')
    estab, curvas_estab = experimento(tr, te, {
        'DQN completa': {},
        'Sin Experience Replay': {'replay': False},
        'Sin Red Objetivo': {'red_objetivo': False},
    })
    print('Experimento de variables (3 semillas)...')
    variables, _ = experimento(tr, te, {
        'DQN · troponina + CK-MB': {'columnas': MARCADORES},
        'DQN · 8 variables': {'columnas': FEATURES},
    })

    resultados = {
        'n_entrenamiento': len(tr), 'n_prueba': len(te),
        'comparacion': comparacion,
        'curvas_entrenamiento': {qtable.nombre: curva(hist_q['recompensas']), dqn8.nombre: curva(hist_8['recompensas']),
                                 'DQN final · troponina + CK-MB': curva(hist_2['recompensas'])},
        'estabilidad': estab, 'curvas_estabilidad': curvas_estab,
        'variables': variables,
    }
    (CARPETA_RESULTADOS / 'resultados_app.json').write_text(json.dumps(resultados, ensure_ascii=False, indent=1))
    print(f'Listo en {time.time() - t0:.0f} s → modelos/ y resultados/resultados_app.json')


if __name__ == '__main__':
    main()
