"""Centro de Triaje · Agente DQN (Taller 3, Parte 3).

Ejecutar:  streamlit run app.py
"""
import json
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from triaje_rl.core import (ACCIONES, CARPETA_MODELOS, CARPETA_RESULTADOS, COLOR, FEATURES, MARCADORES,
                            R_TABLA, RAIZ, SIGNIFICADO, AgenteDQN, AgenteQTable, cargar_datos, evaluar,
                            frontera, media_movil)

st.set_page_config(page_title='Centro de Triaje · DQN', page_icon='🫀', layout='wide')

RUTA_REGISTRO = RAIZ / 'registros' / 'decisiones.csv'
ESCALA_COLOR = alt.Scale(domain=ACCIONES, range=[COLOR[a] for a in ACCIONES])
NOMBRE_FINAL = 'DQN final · troponina + CK-MB'
NOMBRE_SESION = 'Agente entrenado en esta sesión'

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.titulo {font-size: 1.9rem; font-weight: 700; margin-bottom: 0;}
.subtitulo {color: #8fa3b8; margin-top: 2px; margin-bottom: 14px;}
.kpi {background: #172231; border: 1px solid #23324a; border-radius: 12px; padding: 12px 16px;}
.kpi .v {font-size: 1.6rem; font-weight: 700;}
.kpi .l {color: #8fa3b8; font-size: .82rem;}
.kpi .d {font-size: .8rem; color: #6fd39a;}
.card {background: #172231; border: 1px solid #23324a; border-radius: 16px; padding: 18px 20px;}
.card-head {display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px;}
.card-id {font-size: 1.25rem; font-weight: 700;}
.card-meta {color: #8fa3b8;}
.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 8px; margin-bottom: 14px;}
.grid div {background: #0e1621; border-radius: 10px; padding: 7px 10px;}
.grid label {display: block; color: #8fa3b8; font-size: .72rem;}
.grid b {font-size: 1.02rem;}
.grid .clave {outline: 1px solid #3aa0c8;}
.decision {display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center; background: #0e1621; border-radius: 12px; padding: 14px 16px;}
.decision .qbars {flex: 1 1 240px;}
.sem {background: #05090f; border-radius: 30px; padding: 9px 8px; display: flex; flex-direction: column; gap: 7px;}
.lamp {width: 30px; height: 30px; border-radius: 50%; background: #253142;}
.on-Rojo {background: #d63b3b; box-shadow: 0 0 16px 4px rgba(214,59,59,.6);}
.on-Amarillo {background: #e3a512; box-shadow: 0 0 16px 4px rgba(227,165,18,.6);}
.on-Verde {background: #2e9e5b; box-shadow: 0 0 16px 4px rgba(46,158,91,.6);}
.dec-t {font-size: 1.7rem; font-weight: 800; letter-spacing: .5px;}
.dec-s {color: #b7c4d3;}
.qbar {display: grid; grid-template-columns: 72px 1fr 46px; gap: 8px; align-items: center; font-size: .82rem; margin: 3px 0;}
.qbar .bg {background: #23324a; border-radius: 6px; height: 10px;}
.qbar .fg {height: 10px; border-radius: 6px;}
.foot {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;}
.chip {border-radius: 999px; padding: 4px 12px; font-size: .82rem; background: #23324a;}
.ok {background: rgba(46,158,91,.25); color: #8de0ae;}
.warn {background: rgba(227,165,18,.22); color: #f3cd72;}
.bad {background: rgba(214,59,59,.3); color: #ff9b9b; font-weight: 700;}
.big {padding: 24px 26px;}
.big .lamp {width: 44px; height: 44px;}
.big .dec-t {font-size: 2.5rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ recursos
@st.cache_data
def datos():
    return cargar_datos()


@st.cache_resource
def agentes_base():
    if not (CARPETA_MODELOS / 'dqn_final.pt').exists():
        import entrenar
        with st.spinner('Entrenando los agentes por primera vez (≈ 1 minuto)...'):
            entrenar.main()
    return {
        NOMBRE_FINAL: AgenteDQN.cargar(CARPETA_MODELOS / 'dqn_final.pt'),
        'DQN · 8 variables': AgenteDQN.cargar(CARPETA_MODELOS / 'dqn_8_variables.pt'),
        'Q-Table (Taller 2)': AgenteQTable.cargar(CARPETA_MODELOS / 'qtable_taller2.npz'),
    }


@st.cache_data
def resultados():
    return json.loads((CARPETA_RESULTADOS / 'resultados_app.json').read_text())


df, tr, te = datos()
AGENTES = dict(agentes_base())
if NOMBRE_SESION in st.session_state:
    AGENTES[NOMBRE_SESION] = st.session_state[NOMBRE_SESION]
QTABLE = AGENTES['Q-Table (Taller 2)']


# ------------------------------------------------------------------ utilidades de presentación
def resultado_clinico(accion, infarto):
    return {
        ('Rojo', 1): ('✔ Infarto escalado a tiempo', 'ok'), ('Rojo', 0): ('Falsa alarma', 'warn'),
        ('Amarillo', 1): ('Detectado con demora', 'warn'), ('Amarillo', 0): ('Revisión innecesaria', ''),
        ('Verde', 0): ('✔ Seguimiento correcto', 'ok'), ('Verde', 1): ('⛔ INFARTO NO DETECTADO', 'bad'),
    }[(accion, int(infarto))]


def semaforo(accion):
    return '<div class="sem">' + ''.join(
        f'<div class="lamp {"on-" + a if a == accion else ""}"></div>' for a in ('Rojo', 'Amarillo', 'Verde')) + '</div>'


def barras_q(q):
    lo, hi = min(q), max(q)
    filas = []
    for a, v in zip(ACCIONES, q):
        ancho = 25 + 75 * (v - lo) / (hi - lo + 1e-9)
        filas.append(f'<div class="qbar"><span>{a}</span><div class="bg"><div class="fg" style="width:{ancho:.0f}%;'
                     f'background:{COLOR[a]}"></div></div><span>{v:.1f}</span></div>')
    return '<div class="qbars"><div style="color:#8fa3b8;font-size:.75rem;margin-bottom:4px">Valor Q (recompensa futura esperada)</div>' + ''.join(filas) + '</div>'


def tarjeta_paciente(p, q, accion, extra_chips='', clase=''):
    sexo = 'Hombre' if p.gender == 1 else 'Mujer'
    campos = [('Troponina', f'{p.troponin:g} ng/mL', True), ('CK-MB', f'{p.ck_mb:g} ng/mL', True),
              ('Frecuencia cardíaca', f'{p.heart_rate:.0f} lpm', False),
              ('Presión arterial', f'{p.systolic_bp:.0f}/{p.diastolic_bp:.0f}', False),
              ('Glucosa', f'{p.blood_sugar:.0f} mg/dL', False), ('Edad', f'{p.age:.0f} años', False)]
    grid = ''.join(f'<div class="{"clave" if c else ""}"><label>{l}</label><b>{v}</b></div>' for l, v, c in campos)
    ident = f'Paciente #{int(p.paciente)}' if 'paciente' in p else 'Paciente en evaluación'
    return f"""
    <div class="card {clase}">
      <div class="card-head"><span class="card-id">{ident}</span><span class="card-meta">{sexo} · {p.age:.0f} años</span></div>
      <div class="grid">{grid}</div>
      <div class="decision">{semaforo(accion)}
        <div style="min-width:170px"><div class="dec-t" style="color:{COLOR[accion]}">{accion.upper()}</div>
        <div class="dec-s">{SIGNIFICADO[accion]}</div></div>{barras_q(q)}</div>
      <div class="foot">{extra_chips}</div>
    </div>"""


def kpi(col, valor, etiqueta, detalle=''):
    col.markdown(f'<div class="kpi"><div class="v">{valor}</div><div class="l">{etiqueta}</div>'
                 f'<div class="d">{detalle}</div></div>', unsafe_allow_html=True)


def grafico_lineas(dfl, x, y, color, titulo, alto=210, escala_y=None, dominio=None):
    enc_color = alt.Color(f'{color}:N', legend=alt.Legend(orient='bottom', title=None),
                          scale=alt.Scale(domain=list(dominio), range=list(dominio.values())) if dominio else alt.Undefined)
    return (alt.Chart(dfl, title=titulo).mark_line(strokeWidth=2.2)
            .encode(x=alt.X(f'{x}:Q', title=None),
                    y=alt.Y(f'{y}:Q', title=None, scale=escala_y or alt.Scale(zero=False)), color=enc_color)
            .properties(height=alto))


def mapa_decisiones(agente, titulo, puntos=None, alto=360, extra=None):
    f = frontera(agente, n=70)
    paso_t, paso_c = np.diff(np.unique(f.log_trop))[0], np.diff(np.unique(f.log_ck))[0]
    f['x2'], f['y2'] = f.log_trop + paso_t, f.log_ck + paso_c
    fondo = alt.Chart(f).mark_rect(opacity=0.55).encode(
        x=alt.X('log_trop:Q', title='log10 Troponina', scale=alt.Scale(domain=[-3.1, 1.2], nice=False)),
        x2='x2', y=alt.Y('log_ck:Q', title='log10 CK-MB', scale=alt.Scale(domain=[-0.6, 2.7], nice=False)), y2='y2',
        color=alt.Color('decision:N', scale=ESCALA_COLOR, legend=alt.Legend(title='Decisión', orient='bottom')))
    capas = [fondo]
    if puntos is not None:
        capas.append(alt.Chart(puntos).mark_point(filled=True, size=34, opacity=0.9, strokeWidth=1).encode(
            x='log_trop:Q', y='log_ck:Q',
            shape=alt.Shape('Resultado real:N', scale=alt.Scale(domain=['Sano', 'Infarto'], range=['circle', 'cross']),
                            legend=alt.Legend(orient='bottom')),
            color=alt.value('#f5f7fa'), tooltip=['Paciente', 'Resultado real', 'Troponina', 'CK-MB']))
    if extra is not None:
        capas.append(extra)
    st.markdown(f'**{titulo}**')
    return alt.layer(*capas).properties(height=alto, padding={'left': 14, 'top': 4, 'right': 8, 'bottom': 4})


def puntos_prueba():
    return pd.DataFrame({'log_trop': np.log10(te.troponin), 'log_ck': np.log10(te.ck_mb),
                         'Resultado real': np.where(te.infarto == 1, 'Infarto', 'Sano'),
                         'Paciente': te.paciente, 'Troponina': te.troponin, 'CK-MB': te.ck_mb})


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown('### 🫀 Centro de Triaje')
    opciones = list(AGENTES)
    agente_nombre = st.selectbox('Agente activo', [o for o in opciones if o != 'Q-Table (Taller 2)'],
                                 index=0, help='El agente que decide en Urgencias y en la evaluación de pacientes.')
    AGENTE = AGENTES[agente_nombre]
    st.caption(f'Variables que observa: {", ".join(getattr(AGENTE, "columnas", ["troponin"]))}')
    st.divider()
    st.markdown('**Matriz de recompensa (Taller 2)**')
    st.dataframe(pd.DataFrame(R_TABLA[::-1], index=['Infarto', 'Sano'], columns=ACCIONES), width='stretch')
    st.markdown('**Semáforo**')
    for a in ('Rojo', 'Amarillo', 'Verde'):
        st.markdown(f'<span style="color:{COLOR[a]}">●</span> **{a}**: {SIGNIFICADO[a].lower()}', unsafe_allow_html=True)
    st.divider()
    st.caption('Prototipo académico de apoyo a la decisión. No es un dispositivo médico ni reemplaza el criterio clínico.')

# ------------------------------------------------------------------ encabezado
st.markdown('<div class="titulo">Centro de Triaje · Agente de Aprendizaje por Refuerzo</div>'
            '<div class="subtitulo">Sistema de Predicción de Enfermedad Cardíaca · del Q-Learning tabular (Taller 2) a una '
            'Deep Q-Network (Taller 3)</div>', unsafe_allow_html=True)

RES = resultados()
fin, base = RES['comparacion'][NOMBRE_FINAL], RES['comparacion']['Q-Table (Taller 2)']
k = st.columns(4)
kpi(k[0], f'{fin["Recompensa clínica media"]:.2f}', 'Recompensa clínica media (DQN final)',
    f'Q-Table del Taller 2: {base["Recompensa clínica media"]:.2f}')
kpi(k[1], fin['Falsas alarmas'], f'Falsas alarmas en {RES["n_prueba"]} pacientes nuevos', f'Q-Table: {base["Falsas alarmas"]}')
kpi(k[2], fin['Infartos en Verde'], 'Infartos enviados a Verde', f'Q-Table: {base["Infartos en Verde"]}')
kpi(k[3], f'{fin["Sanos en Verde"]:.0%}', 'Sanos liberados a seguimiento', f'Q-Table: {base["Sanos en Verde"]:.0%}')
st.write('')

tab_urg, tab_pac, tab_ent, tab_res = st.tabs(['🚑 Urgencias en vivo', '🩺 Evaluar paciente',
                                              '🧠 Entrenamiento en vivo', '📊 Resultados del experimento'])

# ================================================================== 1. URGENCIAS EN VIVO
with tab_urg:
    st.markdown('Llegan pacientes **que el agente nunca vio**. El agente activo decide el semáforo en tiempo real; '
                'en paralelo, la Q-Table del Taller 2 decide sobre los mismos pacientes.')
    c = st.columns([1.3, 1.3, 1, 1])
    n_pac = c[0].slider('Pacientes del turno', 20, len(te), 80, 10)
    velocidad = c[1].select_slider('Velocidad', ['Lenta', 'Normal', 'Rápida', 'Muy rápida'], value='Normal')
    st.session_state.setdefault('turno', 0)
    c[2].write(''); c[3].write('')
    iniciar = c[2].button('▶ Iniciar turno', type='primary', width='stretch')
    if c[3].button('↺ Limpiar', width='stretch'):
        st.session_state.pop('ultimo_turno', None)

    izq, der = st.columns([1.45, 1])
    ph_card, ph_tabla = izq.empty(), izq.empty()
    ph_kpi, ph_chart, ph_alertas = der.empty(), der.empty(), der.empty()

    def pintar_turno(reg, i, criticos):
        r = reg.iloc[i]
        p = te.loc[r.idx]
        txt, clase = resultado_clinico(r.dqn, r.real)
        txt_q, clase_q = resultado_clinico(r.qtab, r.real)
        chips = (f'<span class="chip">Q-Table (Taller 2): <b style="color:{COLOR[r.qtab]}">{r.qtab}</b></span>'
                 f'<span class="chip">Resultado real: <b>{"INFARTO" if r.real else "sano"}</b></span>'
                 f'<span class="chip {clase}">Agente: {txt}</span><span class="chip {clase_q}">Q-Table: {txt_q}</span>')
        q = [r.q_verde, r.q_amarillo, r.q_rojo]
        ph_card.markdown(tarjeta_paciente(p, q, r.dqn, chips), unsafe_allow_html=True)

        hecho = reg.iloc[:i + 1]
        with ph_kpi.container():
            st.markdown(f'**Turno {st.session_state.turno}** · paciente {i + 1} de {len(reg)}')
            filas = {}
            for nombre, col in (('Agente activo', 'dqn'), ('Q-Table (Taller 2)', 'qtab')):
                a, real = hecho[col], hecho.real
                filas[nombre] = {'Recompensa acumulada': f'{hecho[f"r_{col}"].sum():.0f}',
                                 'Falsas alarmas': int(((a == 'Rojo') & (real == 0)).sum()),
                                 'Infartos enviados a Verde': int(((a == 'Verde') & (real == 1)).sum()),
                                 'Sanos liberados a Verde': int(((a == 'Verde') & (real == 0)).sum())}
            st.dataframe(pd.DataFrame(filas).astype(str), width='stretch')
        acum = pd.DataFrame({'Paciente': np.arange(1, i + 2),
                             'Agente activo': hecho.r_dqn.cumsum(), 'Q-Table (Taller 2)': hecho.r_qtab.cumsum()})
        acum = acum.melt('Paciente', var_name='Agente', value_name='Recompensa acumulada')
        ph_chart.altair_chart(grafico_lineas(acum, 'Paciente', 'Recompensa acumulada', 'Agente',
                                             'Recompensa clínica acumulada', 230, alt.Scale(zero=True),
                                             {'Agente activo': '#3aa0c8', 'Q-Table (Taller 2)': '#e3a512'}),
                              width='stretch')
        ult = hecho.tail(10).iloc[::-1]
        tabla = pd.DataFrame({'Paciente': ult.paciente, 'Troponina': ult.troponina, 'CK-MB': ult.ck_mb,
                              'Agente': ult.dqn, 'Q-Table': ult.qtab,
                              'Real': np.where(ult.real == 1, 'Infarto', 'Sano')})
        pinta = lambda v: f'color:{COLOR[v]};font-weight:700' if v in COLOR else ''
        ph_tabla.dataframe(tabla.style.map(pinta, subset=['Agente', 'Q-Table']).format(
            {'Troponina': '{:g}', 'CK-MB': '{:g}'}), width='stretch', hide_index=True)
        with ph_alertas.container():
            if criticos:
                st.error('**Casos críticos del agente** (infarto enviado a Verde): '
                         + ', '.join(f'#{c}' for c in criticos) + ' → requieren revisión humana.')
            else:
                st.success('Sin infartos enviados a Verde en este turno.')

    if iniciar:
        st.session_state.turno += 1
        orden = te.sample(n=n_pac, random_state=st.session_state.turno)
        q_dqn = AGENTE.q_valores(orden)
        reg = pd.DataFrame({
            'idx': orden.index, 'paciente': orden.paciente.to_numpy(), 'troponina': orden.troponin.to_numpy(),
            'ck_mb': orden.ck_mb.to_numpy(), 'real': orden.infarto.to_numpy(),
            'dqn': np.array(ACCIONES)[q_dqn.argmax(1)], 'qtab': np.array(ACCIONES)[QTABLE.decidir(orden)],
            'q_verde': q_dqn[:, 0], 'q_amarillo': q_dqn[:, 1], 'q_rojo': q_dqn[:, 2]})
        reg['r_dqn'] = R_TABLA[reg.real, [ACCIONES.index(a) for a in reg.dqn]]
        reg['r_qtab'] = R_TABLA[reg.real, [ACCIONES.index(a) for a in reg.qtab]]
        pausa = {'Lenta': 1.4, 'Normal': 0.6, 'Rápida': 0.18, 'Muy rápida': 0.02}[velocidad]
        criticos = []
        for i in range(len(reg)):
            r = reg.iloc[i]
            if r.dqn == 'Verde' and r.real == 1:
                criticos.append(int(r.paciente))
                st.toast(f'⛔ Paciente #{int(r.paciente)}: infarto enviado a Verde. Requiere revisión humana.')
            pintar_turno(reg, i, criticos)
            time.sleep(pausa)
        st.session_state.ultimo_turno = (reg, criticos)
        dif = reg.r_dqn.sum() - reg.r_qtab.sum()
        st.toast(f'Turno terminado · el agente obtuvo {dif:+.0f} puntos frente a la Q-Table')
    elif 'ultimo_turno' in st.session_state:
        reg, criticos = st.session_state.ultimo_turno
        pintar_turno(reg, len(reg) - 1, criticos)
    else:
        ph_card.info('Presiona **▶ Iniciar turno** para que empiecen a llegar pacientes.')

# ================================================================== 2. EVALUAR PACIENTE
PRESETS = {
    'Caso ambiguo del mockup': dict(age=49, gender=0, heart_rate=67, systolic_bp=114, diastolic_bp=70,
                                    blood_sugar=96.0, troponin=0.005, ck_mb=0.676),
    'Infarto típico': dict(age=64, gender=1, heart_rate=88, systolic_bp=150, diastolic_bp=92,
                           blood_sugar=160.0, troponin=0.693, ck_mb=2.65),
    'Paciente sano': dict(age=45, gender=0, heart_rate=72, systolic_bp=120, diastolic_bp=78,
                          blood_sugar=105.0, troponin=0.003, ck_mb=1.99),
}
fila_mockup = df[(df.troponin == 0.005) & (df.ck_mb == 0.676)].iloc[0]
ENTEROS = {'age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp'}
PRESETS['Caso ambiguo del mockup'].update(
    {c: int(fila_mockup[c]) if c in ENTEROS else float(fila_mockup[c]) for c in FEATURES})


def cargar_preset(nombre):
    for c, v in PRESETS[nombre].items():
        st.session_state[f'p_{c}'] = v
    st.session_state.preset = nombre


for c, v in PRESETS['Paciente sano'].items():
    st.session_state.setdefault(f'p_{c}', v)

with tab_pac:
    st.markdown('Flujo del mockup: **1** datos del paciente → **2** decisión del agente → **3** validación e intervención humana.')
    b = st.columns(3)
    for col, nombre in zip(b, PRESETS):
        col.button(f'Cargar: {nombre}', on_click=cargar_preset, args=(nombre,), width='stretch')

    c1, c2, c3 = st.columns([1, 1.6, 1.15])
    with c1:
        st.markdown('#### 1 · Datos del paciente')
        st.number_input('Troponina (ng/mL)', 0.001, 11.0, step=0.001, format='%.3f', key='p_troponin')
        st.number_input('CK-MB (ng/mL)', 0.3, 300.0, step=0.01, format='%.3f', key='p_ck_mb')
        st.number_input('Edad', 14, 103, key='p_age')
        st.radio('Sexo', [1, 0], format_func=lambda x: 'Hombre' if x == 1 else 'Mujer', horizontal=True, key='p_gender')
        with st.expander('Signos vitales y glucosa'):
            st.number_input('Frecuencia cardíaca (lpm)', 30, 250, key='p_heart_rate')
            st.number_input('Presión sistólica', 40, 230, key='p_systolic_bp')
            st.number_input('Presión diastólica', 30, 160, key='p_diastolic_bp')
            st.number_input('Glucosa (mg/dL)', 35.0, 550.0, step=1.0, key='p_blood_sugar')

    paciente = pd.Series({c: st.session_state[f'p_{c}'] for c in FEATURES})
    d = paciente.to_frame().T.astype(float)
    q = AGENTE.q_valores(d)[0]
    decision = ACCIONES[int(q.argmax())]
    orden_q = np.sort(q)[::-1]
    margen = orden_q[0] - orden_q[1]

    with c2:
        st.markdown(f'#### 2 · Decisión del agente · {agente_nombre}')
        st.markdown(tarjeta_paciente(paciente, q, decision, clase='big'), unsafe_allow_html=True)
        if margen < 5:
            st.warning(f'Decisión ajustada: solo {margen:.1f} puntos separan a **{decision}** de la segunda opción. '
                       'Se sugiere revisión clínica.')
        else:
            st.info(f'Margen de decisión: {margen:.1f} puntos sobre la segunda opción.')
        coincide = (df.troponin == paciente.troponin) & (df.ck_mb == paciente.ck_mb)
        if coincide.any():
            real = df[coincide].iloc[0]
            st.caption(f'Paciente real del dataset (#{int(real.paciente)}): resultado registrado → '
                       f'**{"INFARTO" if real.infarto else "sano"}**.')
        if getattr(AGENTE, 'columnas', None) == MARCADORES:
            punto = alt.Chart(pd.DataFrame({'x': [np.log10(paciente.troponin)], 'y': [np.log10(paciente.ck_mb)]})).mark_point(
                shape='diamond', size=420, filled=True, color='#ffffff', stroke='#000000', strokeWidth=2).encode(x='x:Q', y='y:Q')
            st.altair_chart(mapa_decisiones(AGENTE, 'Ubicación del paciente (◆) en el mapa del agente',
                                            alto=300, extra=punto), width='stretch')

    with c3:
        st.markdown('#### 3 · Validación e intervención humana')
        override = st.toggle('Sobrescribir la decisión del agente')
        final = decision
        razon = ''
        if override:
            final = st.radio('Decisión final', ACCIONES, index=ACCIONES.index('Rojo'), horizontal=True)
            razon = st.text_area('Razón del override (notas médicas)',
                                 placeholder='Ej.: dolor torácico típico y ECG con elevación del ST pese a marcadores normales')
        if st.button('Registrar en la historia clínica', type='primary', width='stretch'):
            if override and not razon.strip():
                st.error('Para sobrescribir la decisión hay que documentar la razón.')
            else:
                RUTA_REGISTRO.parent.mkdir(exist_ok=True)
                fila = pd.DataFrame([{'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'agente': agente_nombre,
                                      'troponina': paciente.troponin, 'ck_mb': paciente.ck_mb,
                                      'decision_agente': decision, 'decision_final': final,
                                      'override': override, 'razon': razon.strip()}])
                fila.to_csv(RUTA_REGISTRO, mode='a', header=not RUTA_REGISTRO.exists(), index=False)
                if override and final != decision:
                    st.success(f'Override registrado: {decision} → **{final}**. Se notifica al equipo médico.')
                else:
                    st.success(f'Decisión **{final}** confirmada y registrada.')
        if RUTA_REGISTRO.exists():
            st.markdown('**Últimos registros**')
            st.dataframe(pd.read_csv(RUTA_REGISTRO).tail(6).iloc[::-1][['fecha', 'decision_agente', 'decision_final', 'override']],
                         width='stretch', hide_index=True)

# ================================================================== 3. ENTRENAMIENTO EN VIVO
with tab_ent:
    st.markdown('Entrena una DQN desde cero y observa cómo aprende. Prueba a **quitar los trucos de estabilidad** '
                'o a cambiar las variables que ve el agente.')
    c = st.columns([1.4, 1, 1, 1.2, 0.8])
    vars_sel = c[0].radio('Variables que observa', ['Troponina + CK-MB', '8 variables'], horizontal=True)
    replay = c[1].toggle('Experience Replay', True)
    objetivo = c[2].toggle('Red Objetivo', True)
    pasos = c[3].slider('Pacientes de entrenamiento', 2000, 15000, 10000, 1000)
    semilla = c[4].number_input('Semilla', 0, 99, 0)
    entrenar_btn = st.button('🧠 Entrenar agente', type='primary')

    ph_prog = st.empty()
    g = st.columns(3)
    ph_g = [g[i].empty() for i in range(3)]
    ph_res = st.container()

    def pintar_curvas(hist, n_total):
        v = 200
        rec = media_movil(hist['recompensas'], v)
        x = np.arange(len(rec))
        datos_r = pd.DataFrame({'Paso': x[v::25], 'Recompensa': rec[v::25]})
        ph_g[0].altair_chart(alt.Chart(datos_r, title='Recompensa clínica (media móvil)').mark_line(color='#3aa0c8', strokeWidth=2.2)
                             .encode(x=alt.X('Paso:Q', scale=alt.Scale(domain=[0, n_total]), title=None),
                                     y=alt.Y('Recompensa:Q', scale=alt.Scale(domain=[-8, 9]), title=None)).properties(height=220),
                             width='stretch')
        if len(hist['perdidas']) > v:
            per = media_movil(hist['perdidas'], v)
            qm = media_movil(hist['q_medio'], v)
            xp = np.arange(len(per))
            ph_g[1].altair_chart(alt.Chart(pd.DataFrame({'Paso': xp[v::25], 'Pérdida': per[v::25]}), title='Error de la red (pérdida Huber)')
                                 .mark_line(color='#d63b3b', strokeWidth=2.2)
                                 .encode(x=alt.X('Paso:Q', scale=alt.Scale(domain=[0, n_total]), title=None),
                                         y=alt.Y('Pérdida:Q', scale=alt.Scale(type='log'), title=None)).properties(height=220),
                                 width='stretch')
            ph_g[2].altair_chart(alt.Chart(pd.DataFrame({'Paso': xp[v::25], 'Q': qm[v::25]}), title='Valor Q promedio estimado')
                                 .mark_line(color='#e3a512', strokeWidth=2.2)
                                 .encode(x=alt.X('Paso:Q', scale=alt.Scale(domain=[0, n_total]), title=None),
                                         y=alt.Y('Q:Q', scale=alt.Scale(domain=[-10, 100]), title=None)).properties(height=220),
                                 width='stretch')

    if entrenar_btn:
        columnas = MARCADORES if vars_sel == 'Troponina + CK-MB' else FEATURES
        nuevo = AgenteDQN(columnas)

        def progreso(t, hist):
            eps = hist['epsilon'][-1]
            ph_prog.progress(t / pasos, text=f'Paciente {t:,} de {pasos:,} · exploración ε = {eps:.2f} · '
                                             f'{"explorando" if eps > 0.3 else "explotando lo aprendido"}')
            pintar_curvas(hist, pasos)
            time.sleep(0.12)   # pausa para que el aprendizaje se pueda seguir a simple vista

        hist = nuevo.entrenar(tr, pasos=pasos, replay=replay, red_objetivo=objetivo, seed=int(semilla), callback=progreso)
        ev = evaluar(nuevo.decidir(te), te)
        st.session_state[NOMBRE_SESION] = nuevo
        st.session_state.ultimo_entrenamiento = (hist, ev, columnas, replay, objetivo, pasos)

    if 'ultimo_entrenamiento' in st.session_state:
        hist, ev, columnas, rep, obj, n_total = st.session_state.ultimo_entrenamiento
        if not entrenar_btn:
            pintar_curvas(hist, n_total)
        with ph_res:
            st.markdown(f'#### Resultado en {len(te)} pacientes nuevos · '
                        f'{"troponina + CK-MB" if columnas == MARCADORES else "8 variables"} · '
                        f'replay {"✔" if rep else "✘"} · red objetivo {"✔" if obj else "✘"}')
            k = st.columns(4)
            kpi(k[0], f'{ev["Recompensa clínica media"]:.2f}', 'Recompensa clínica media', f'Q-Table: {base["Recompensa clínica media"]:.2f}')
            kpi(k[1], ev['Falsas alarmas'], 'Falsas alarmas', f'Q-Table: {base["Falsas alarmas"]}')
            kpi(k[2], ev['Infartos en Verde'], 'Infartos enviados a Verde', f'Q-Table: {base["Infartos en Verde"]}')
            kpi(k[3], f'{ev["Sanos en Verde"]:.0%}', 'Sanos liberados a seguimiento', f'Q-Table: {base["Sanos en Verde"]:.0%}')
            st.caption(f'Este agente quedó disponible como **“{NOMBRE_SESION}”** en el selector de la barra lateral, '
                       'para usarlo en Urgencias y en la evaluación de pacientes.')

# ================================================================== 4. RESULTADOS DEL EXPERIMENTO
with tab_res:
    st.markdown(f'Resultados del script `entrenar.py` · {RES["n_entrenamiento"]} pacientes de entrenamiento, '
                f'{RES["n_prueba"]} de prueba (nunca vistos).')
    comp = pd.DataFrame(RES['comparacion']).T.astype(float)
    orden_nombres = list(comp.index)
    paleta = dict(zip(orden_nombres, ['#7b8794', '#e3a512', '#7fb3c8', '#3aa0c8']))
    cbar = st.columns(3)
    for col, metrica in zip(cbar, ['Recompensa clínica media', 'Falsas alarmas', 'Infartos en Verde']):
        dd = comp[[metrica]].reset_index(names='Política')
        barras = alt.Chart(dd, title=metrica).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X('Política:N', sort=orden_nombres, title=None, axis=alt.Axis(labelAngle=0, labelLimit=95)),
            y=alt.Y(f'{metrica}:Q', title=None),
            color=alt.Color('Política:N', scale=alt.Scale(domain=orden_nombres, range=list(paleta.values())), legend=None))
        texto = barras.mark_text(dy=-8, color='#e6edf3').encode(text=alt.Text(f'{metrica}:Q', format='.2f' if 'media' in metrica else 'd'))
        col.altair_chart((barras + texto).properties(height=280), width='stretch')

    tabla = comp.copy()
    for c_ in ['Infartos escalados a Rojo', 'Sanos en Verde', '% Rojo', '% Amarillo', '% Verde']:
        tabla[c_] = (tabla[c_] * 100).round(0).astype(int).astype(str) + ' %'
    tabla['Recompensa clínica media'] = tabla['Recompensa clínica media'].map('{:.2f}'.format)
    st.dataframe(tabla, width='stretch')

    st.markdown('#### Mapa de decisiones sobre los pacientes de prueba')
    st.caption('Fondo: lo que decide cada agente en cada combinación de marcadores. Puntos: pacientes reales (● sano, ✚ infarto). '
               'La Q-Table solo puede decidir por franjas de troponina; la DQN traza su propia frontera con los dos marcadores.')
    m = st.columns(2)
    with m[0]:
        st.altair_chart(mapa_decisiones(QTABLE, 'Q-Table (Taller 2)', puntos_prueba()), width='stretch')
    with m[1]:
        st.altair_chart(mapa_decisiones(AGENTES[NOMBRE_FINAL], NOMBRE_FINAL, puntos_prueba()), width='stretch')

    st.markdown('#### Experimento 1 · ¿Por qué hacen falta los trucos de estabilidad?')
    estab = pd.DataFrame(RES['estabilidad']).set_index('Variante')
    st.dataframe(estab.style.format({'Recompensa media': '{:.2f}', 'Recompensa desv.': '± {:.2f}',
                                     'Infartos en Verde (prom.)': '{:.1f}', 'Falsas alarmas (prom.)': '{:.1f}'}), width='stretch')
    colores_var = {'DQN completa': '#3aa0c8', 'Sin Experience Replay': '#d63b3b', 'Sin Red Objetivo': '#e3a512'}
    ce = st.columns(3)
    for col, (clave, titulo) in zip(ce, [('recompensa', 'Recompensa durante el entrenamiento'),
                                          ('perdida', 'Error de la red (pérdida Huber)'),
                                          ('q_medio', 'Valor Q promedio estimado')]):
        filas = []
        for nombre, cur in RES['curvas_estabilidad'].items():
            ys = np.array([s['y'] for s in cur[clave]]).mean(0)
            filas.append(pd.DataFrame({'Paso': cur[clave][0]['x'], 'Valor': ys, 'Variante': nombre}))
        escala = alt.Scale(type='log') if clave == 'perdida' else alt.Scale(zero=False)
        col.altair_chart(grafico_lineas(pd.concat(filas), 'Paso', 'Valor', 'Variante', titulo, 250, escala, colores_var),
                         width='stretch')
    st.caption('Promedio de 3 semillas. Sin Experience Replay el agente aprende de pacientes consecutivos (datos correlacionados): '
               'el error no baja y la política se vuelve inestable. Sin Red Objetivo los valores Q se disparan al inicio '
               '(persiguen su propio objetivo), aunque en este entorno el ranking de acciones casi no cambia porque el siguiente '
               'paciente no depende de la decisión.')

    st.markdown('#### Experimento 2 · ¿Ver más variables ayuda?')
    st.dataframe(pd.DataFrame(RES['variables']).set_index('Variante').style.format(
        {'Recompensa media': '{:.2f}', 'Recompensa desv.': '± {:.2f}', 'Infartos en Verde (prom.)': '{:.1f}',
         'Falsas alarmas (prom.)': '{:.1f}'}), width='stretch')
    st.caption('Con ~1 000 pacientes, las 6 variables extra solo agregan ruido: troponina y CK-MB concentran casi toda la '
               'información (≈ 98 % de la importancia en el AdaBoost del proyecto). La mejora viene de ver los marcadores '
               'continuos, no de ver más variables.')
