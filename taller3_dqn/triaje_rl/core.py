"""Datos, entorno y agentes del triaje de infarto.

El problema es el del Taller 2: en cada paso llega un paciente, el agente elige un color del semáforo
(Verde / Amarillo / Rojo), recibe la recompensa de la matriz clínica según el resultado real y el siguiente
estado es el próximo paciente que llega.
"""
import copy
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / 'Medicaldataset.csv'
CARPETA_MODELOS = RAIZ / 'modelos'
CARPETA_RESULTADOS = RAIZ / 'resultados'

SEED = 42
ACCIONES = ['Verde', 'Amarillo', 'Rojo']
VERDE, AMARILLO, ROJO = 0, 1, 2
COLOR = {'Verde': '#2e9e5b', 'Amarillo': '#e3a512', 'Rojo': '#d63b3b'}
SIGNIFICADO = {
    'Verde': 'Seguimiento habitual',
    'Amarillo': 'Repetir medición y valoración clínica',
    'Rojo': 'Escalar de inmediato al equipo médico',
}
# R_TABLA[infarto, acción] — matriz clínica del Taller 2
R_TABLA = np.array([[5.0, 0.5, -3.0],      # sano
                    [-50.0, 2.0, 10.0]])   # infarto

# Hiperparámetros del Taller 2
GAMMA, EPS_INI, EPS_MIN, DECAY, PASOS = 0.9, 1.0, 0.01, 0.999, 10000

FEATURES = ['age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'blood_sugar', 'ck_mb', 'troponin']
MARCADORES = ['troponin', 'ck_mb']
LOG_COLS = ['blood_sugar', 'ck_mb', 'troponin']
COLUMNAS_CSV = {
    'Age': 'age', 'Gender': 'gender', 'Heart rate': 'heart_rate',
    'Systolic blood pressure': 'systolic_bp', 'Diastolic blood pressure': 'diastolic_bp',
    'Blood sugar': 'blood_sugar', 'CK-MB': 'ck_mb', 'Troponin': 'troponin', 'Result': 'result',
}


def fijar_semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def cargar_datos(ruta=RUTA_CSV):
    """Limpieza del Taller 1 y separación 80/20 conservando el orden de llegada."""
    df = pd.read_csv(ruta).rename(columns=COLUMNAS_CSV)
    df = df[(df.heart_rate >= 30) & (df.heart_rate <= 250)].reset_index(drop=True)
    df['infarto'] = (df.result == 'positive').astype(int)
    df['paciente'] = df.index + 1
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=df.infarto, random_state=SEED)
    tr = df.loc[np.sort(idx_tr)].reset_index(drop=True)
    te = df.loc[np.sort(idx_te)].reset_index(drop=True)
    return df, tr, te


def evaluar(acciones, d):
    """Métricas del Taller 2 sobre un conjunto de pacientes."""
    acciones = np.asarray(acciones)
    inf = d.infarto.to_numpy() == 1
    rec = R_TABLA[d.infarto.to_numpy(), acciones]
    return {
        'Recompensa clínica media': float(rec.mean()),
        'Infartos en Verde': int(((acciones == VERDE) & inf).sum()),
        'Falsas alarmas': int(((acciones == ROJO) & ~inf).sum()),
        'Infartos escalados a Rojo': float((acciones[inf] == ROJO).mean()),
        'Sanos en Verde': float((acciones[~inf] == VERDE).mean()),
        '% Rojo': float((acciones == ROJO).mean()),
        '% Amarillo': float((acciones == AMARILLO).mean()),
        '% Verde': float((acciones == VERDE).mean()),
    }


class AgenteQTable:
    """Agente del Taller 2: estado = nivel de troponina (3 cuantiles), Q-Table 3×3."""
    nombre = 'Q-Table (Taller 2)'
    NIVELES = ['Troponina baja', 'Troponina media', 'Troponina alta']

    def __init__(self, tr):
        _, cortes = pd.qcut(tr.troponin, q=3, retbins=True)
        self.cortes = cortes[1:-1]
        self.Q = np.zeros((3, 3))

    def estado(self, d):
        return np.digitize(d.troponin, self.cortes)

    def entrenar(self, tr, alpha=0.1, seed=SEED):
        rng = np.random.default_rng(seed)
        estados, infartos = self.estado(tr), tr.infarto.to_numpy()
        siguientes = np.roll(estados, -1)[:-1]
        estados, infartos = estados[:-1], infartos[:-1]
        grupos = {s: np.flatnonzero(estados == s) for s in np.unique(estados)}
        eps, recompensas = EPS_INI, []
        for _ in range(PASOS):
            s = estados[rng.integers(len(estados))]
            Qs = self.Q[s]
            a = int(rng.integers(3)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Qs == Qs.max())))
            i = grupos[s][rng.integers(len(grupos[s]))]
            r, s2 = R_TABLA[infartos[i], a], siguientes[i]
            self.Q[s, a] += alpha * (r + GAMMA * self.Q[s2].max() - self.Q[s, a])
            recompensas.append(r)
            eps = max(EPS_MIN, eps * DECAY)
        return {'recompensas': np.array(recompensas)}

    def q_valores(self, d):
        return self.Q[self.estado(d)]

    def decidir(self, d):
        return self.q_valores(d).argmax(1)

    def guardar(self, ruta):
        np.savez(ruta, Q=self.Q, cortes=self.cortes)

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(ruta)
        agente = cls.__new__(cls)
        agente.Q, agente.cortes = datos['Q'], datos['cortes']
        return agente


class QNetwork(nn.Module):
    def __init__(self, n_entrada, n_acciones=3, oculta=64):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entrada, oculta), nn.ReLU(),
            nn.Linear(oculta, oculta), nn.ReLU(),
            nn.Linear(oculta, n_acciones),
        )

    def forward(self, s):
        return self.red(s)


class AgenteDQN:
    """Agente del Taller 3: la Q-Table se reemplaza por una red neuronal sobre variables continuas."""

    def __init__(self, columnas=MARCADORES, oculta=64, escala_r=10.0):
        self.columnas, self.oculta, self.escala_r = list(columnas), oculta, escala_r
        self.red = QNetwork(len(self.columnas), oculta=oculta)
        self.mu = self.sd = None

    @property
    def nombre(self):
        return 'DQN · troponina + CK-MB' if self.columnas == MARCADORES else f'DQN · {len(self.columnas)} variables'

    def _crudo(self, d):
        Z = d[self.columnas].astype(float).copy()
        for c in LOG_COLS:
            if c in Z:
                Z[c] = np.log10(Z[c])
        return Z.to_numpy()

    def codificar(self, d):
        return ((self._crudo(d) - self.mu) / self.sd).astype(np.float32)

    def q_valores(self, d):
        with torch.no_grad():
            return self.red(torch.from_numpy(self.codificar(d))).numpy() * self.escala_r

    def decidir(self, d):
        return self.q_valores(d).argmax(1)

    def entrenar(self, tr, pasos=PASOS, replay=True, red_objetivo=True, seed=SEED, lote=64, lr=1e-3,
                 sincronizar=250, capacidad=10000, callback=None, cada=250):
        fijar_semilla(seed)
        rng = np.random.default_rng(seed)
        crudo = self._crudo(tr)
        self.mu, self.sd = crudo.mean(0), crudo.std(0)
        S_all, inf_all = self.codificar(tr), tr.infarto.to_numpy()
        n = len(S_all) - 1                                   # el último paciente no tiene "siguiente"

        self.red = QNetwork(len(self.columnas), oculta=self.oculta)
        q_obj = copy.deepcopy(self.red)
        opt = torch.optim.Adam(self.red.parameters(), lr=lr)
        # Sin replay: solo se recuerdan las últimas `lote` experiencias, en orden de llegada
        memoria = deque(maxlen=capacidad if replay else lote)
        eps = EPS_INI
        hist = {'recompensas': [], 'perdidas': [], 'q_medio': [], 'epsilon': []}

        for t in range(pasos):
            i = int(rng.integers(n))
            s, s2 = S_all[i], S_all[i + 1]
            if rng.random() < eps:
                a = int(rng.integers(3))
            else:
                with torch.no_grad():
                    a = int(self.red(torch.from_numpy(s)).argmax())
            r = R_TABLA[inf_all[i], a]
            memoria.append((s, a, r / self.escala_r, s2))
            hist['recompensas'].append(r)
            hist['epsilon'].append(eps)
            eps = max(EPS_MIN, eps * DECAY)

            if len(memoria) >= lote:
                muestra = [memoria[j] for j in rng.integers(len(memoria), size=lote)] if replay else list(memoria)
                S = torch.from_numpy(np.stack([m[0] for m in muestra]))
                A = torch.tensor([m[1] for m in muestra])
                R = torch.tensor([m[2] for m in muestra], dtype=torch.float32)
                S2 = torch.from_numpy(np.stack([m[3] for m in muestra]))
                q_sa = self.red(S).gather(1, A[:, None]).squeeze(1)
                with torch.no_grad():
                    red = q_obj if red_objetivo else self.red   # sin red objetivo: el objetivo se mueve en cada paso
                    objetivo = R + GAMMA * red(S2).max(1).values
                perdida = F.smooth_l1_loss(q_sa, objetivo)
                opt.zero_grad(); perdida.backward()
                nn.utils.clip_grad_norm_(self.red.parameters(), 10)
                opt.step()
                hist['perdidas'].append(perdida.item())
                hist['q_medio'].append(q_sa.mean().item() * self.escala_r)

            if red_objetivo and t % sincronizar == 0:
                q_obj.load_state_dict(self.red.state_dict())
            if callback and (t + 1) % cada == 0:
                callback(t + 1, hist)

        return {k: np.array(v) for k, v in hist.items()}

    def guardar(self, ruta):
        torch.save({'estado': self.red.state_dict(), 'columnas': self.columnas, 'oculta': self.oculta,
                    'escala_r': self.escala_r, 'mu': torch.tensor(self.mu), 'sd': torch.tensor(self.sd)}, ruta)

    @classmethod
    def cargar(cls, ruta):
        datos = torch.load(ruta, weights_only=True)
        agente = cls(datos['columnas'], datos['oculta'], datos['escala_r'])
        agente.red.load_state_dict(datos['estado'])
        agente.mu, agente.sd = datos['mu'].numpy(), datos['sd'].numpy()
        return agente


def frontera(agente, n=120, rango_t=(-3.1, 1.2), rango_ck=(-0.6, 2.7)):
    """Decisión del agente en una grilla de (log10 troponina, log10 CK-MB). Solo para agentes de 2 marcadores."""
    lt, lc = np.meshgrid(np.linspace(*rango_t, n), np.linspace(*rango_ck, n))
    d = pd.DataFrame({'troponin': 10 ** lt.ravel(), 'ck_mb': 10 ** lc.ravel()})
    return pd.DataFrame({'log_trop': lt.ravel(), 'log_ck': lc.ravel(),
                         'decision': np.array(ACCIONES)[agente.decidir(d)]})


def media_movil(v, n=300):
    return pd.Series(v).rolling(n).mean().to_numpy()
