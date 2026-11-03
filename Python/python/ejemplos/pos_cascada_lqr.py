## pos_cascada_lqr.py
# Control de POSICION del disco del Qube-Servo 3 (Python / pal) con tres algoritmos
# mas avanzados que el PD, con la misma referencia (cuadrada +-1.5 rad, periodo 2.5 s),
# el mismo tiempo de prueba y el mismo formato de CSV que pd_control_qs3_position.py,
# de modo que promediar_resultados.py los compara directamente con el PD:
#
#   cascada : lazo externo P de posicion -> referencia de velocidad -> lazo interno PI de velocidad
#                 w_ref = sat( kp*(r - pos) )          u = sat_10V( kpv*(w_ref - vel) + kiv*integral )
#   lqr     : realimentacion de estado optima (modelo de primer orden del motor)
#                 u = sat_10V( K0*(r - pos) - K1*vel )
#   lqi     : LQR con accion integral (estado extra z = integral del error)
#                 u = sat_10V( K0*(r - pos) - K1*vel + K2*z )
#
# Uso:
#   py pos_cascada_lqr.py identificar                        (mide K y tau del motor; hacerlo ANTES de lqr/lqi)
#   py pos_cascada_lqr.py cascada --kp 15 --kpv 0.3 --kiv 1
#   py pos_cascada_lqr.py lqr --qp 50 --qw 0.1 --r 1
#   py pos_cascada_lqr.py lqi --qp 50 --qw 0.1 --qi 5 --r 1
#   Opciones comunes: --rep 3 (repite la prueba 3 veces seguidas), --sin-graficas
#
# Cada prueba guarda pruebas_python/prueba_<algoritmo>_<ganancias>_run<n>.csv con t, r, y, u, t_wall.
# Por ejemplo:
#   prueba_cascada_kp15_kpv0.3_kiv1_run1.csv     prueba_lqr_qp50_qw0.1_r1_run1.csv
# y despues:   py promediar_resultados.py posicion ITAE --mejores
#
# Requisitos: numpy y scipy (pip install scipy; solo lo usan lqr/lqi/identificar).

import argparse
import json
import os
import signal
import sys
import time
from threading import Thread

import numpy as np
from pal.products.qube import QubeServo3
from pal.utilities.math import SignalGenerator
from pal.utilities.scope import Scope

try:                                  # para imprimir los indices al terminar cada prueba
    from promediar_resultados import metricas_prueba
except ImportError:
    metricas_prueba = None

#region: Configuracion (igual que pd_control_qs3_position.py)
AMPLITUD = 1.5        # rad
PERIODO = 2.5         # s
V_MAX = 10.0          # V
FRECUENCIA = 500      # Hz
TS = 1.0 / FRECUENCIA
STOP_TIME = 9.99      # s

W_MAX = 150.0         # rad/s  limite de la referencia de velocidad en la cascada

# Modelo del disco: w'(t) = (-w + K*u)/tau. Estos valores son TIPICOS y no estan verificados:
# ejecuta 'identificar' y se guardan los reales en modelo_disco.json, que lqr/lqi leen solos.
K_MOTOR_DEF = 23.0    # rad/s por voltio
TAU_DEF = 0.10        # s
MODELO_JSON = 'modelo_disco.json'
CARPETA = 'pruebas_python'
#endregion

# Hilo para poder matar el bucle de control con Ctrl-C
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)


#region: Modelo y ganancias LQR
def cargar_modelo():
    """(K, tau, medido). Si no hay modelo_disco.json se usan los valores tipicos."""
    try:
        with open(MODELO_JSON, encoding='utf-8') as fh:
            d = json.load(fh)
        return float(d['K']), float(d['tau']), True
    except (OSError, KeyError, TypeError, ValueError):
        return K_MOTOR_DEF, TAU_DEF, False


def ganancias_lqr(K, tau, qp, qw, ru, qi=None):
    """Resuelve la ecuacion de Riccati y devuelve (K0, K1, K2) con
         u = K0*(r - pos) - K1*vel + K2*integral(r - pos).
    Estados [pos, vel] (LQR) o [pos, vel, z] con z' = pos - r (LQI).
    Coste J = integral( qp*e^2 + qw*vel^2 (+ qi*z^2) + ru*u^2 ) dt."""
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError:
        sys.exit('Falta scipy para calcular el LQR:  pip install scipy')
    if qi is None:
        A = np.array([[0.0, 1.0], [0.0, -1.0 / tau]])
        B = np.array([[0.0], [K / tau]])
        Q = np.diag([qp, qw])
    else:
        A = np.array([[0.0, 1.0, 0.0], [0.0, -1.0 / tau, 0.0], [1.0, 0.0, 0.0]])
        B = np.array([[0.0], [K / tau], [0.0]])
        Q = np.diag([qp, qw, qi])
    P = solve_continuous_are(A, B, Q, np.array([[ru]]))
    Kmat = (B.T @ P / ru).ravel()                  # u = -Kmat @ [pos - r, vel, z]
    return float(Kmat[0]), float(Kmat[1]), (float(Kmat[2]) if qi is not None else 0.0)
#endregion


#region: Controladores (todos tienen u(r, pos, vel) -> voltaje, y se crean de nuevo en cada prueba)
class Cascada:
    """P de posicion (externo) -> PI de velocidad (interno) con anti-windup condicional."""

    def __init__(self, kp, kpv, kiv):
        self.kp, self.kpv, self.kiv = kp, kpv, kiv
        self.integral = 0.0
        self.etiqueta = f'cascada_kp{kp:g}_kpv{kpv:g}_kiv{kiv:g}'
        self.descripcion = f'Cascada  kp={kp:g}  kpv={kpv:g}  kiv={kiv:g}'

    def u(self, r, pos, vel):
        w_ref = float(np.clip(self.kp * (r - pos), -W_MAX, W_MAX))
        e_v = w_ref - vel
        u_raw = self.kpv * e_v + self.integral
        u = float(np.clip(u_raw, -V_MAX, V_MAX))
        if u == u_raw or e_v * u_raw < 0:          # solo integra si no satura (o si ayuda a salir de la saturacion)
            self.integral += self.kiv * e_v * TS
        return u


class EstadoLQ:
    """Realimentacion de estado u = K0*e - K1*vel + K2*z, con z = integral del error (K2 = 0 -> LQR)."""

    def __init__(self, k0, k1, k2, etiqueta, descripcion):
        self.k0, self.k1, self.k2 = k0, k1, k2
        self.z = 0.0
        self.etiqueta, self.descripcion = etiqueta, descripcion

    def u(self, r, pos, vel):
        e = r - pos
        u_raw = self.k0 * e - self.k1 * vel + self.k2 * self.z
        u = float(np.clip(u_raw, -V_MAX, V_MAX))
        if self.k2 and (u == u_raw or e * u_raw < 0):   # mismo anti-windup condicional
            self.z += e * TS
        return u
#endregion


#region: Bucle de control y resultados
def control_loop(ctrl, log, scopePosition, scopeVoltage):
    squareWave = SignalGenerator().square(AMPLITUD, PERIODO)
    next(squareWave)
    N = int(round(STOP_TIME / TS)) + 1
    t = np.arange(N) * TS
    r, y, u, wall = (np.zeros(N) for _ in range(4))
    done = 0
    with QubeServo3(hardware=1, frequency=FRECUENCIA) as myQube:
        start = time.perf_counter()
        for k in range(N):
            if KILL_THREAD:
                break
            myQube.read_outputs()                       # HIL Read Timebase: espera a la siguiente muestra
            wall[k] = time.perf_counter() - start
            r[k] = squareWave.send(t[k])
            position = float(myQube.motorPosition)      # rad
            speed = float(myQube.motorSpeed)            # rad/s
            u[k] = ctrl.u(r[k], position, speed)
            myQube.write_voltage(u[k])                  # HIL Write
            y[k] = position
            done = k + 1
            if scopePosition is not None:
                scopePosition.sample(t[k], [position, r[k]])
                scopeVoltage.sample(t[k], [u[k]])
    log.update(t=t[:done], r=r[:done], y=y[:done], u=u[:done], wall=wall[:done], completa=(done == N))


def ejecutar(ctrl, scopes):
    """Una prueba completa. Devuelve el log o None si se ha interrumpido."""
    scopePosition = scopeVoltage = None
    if scopes:
        scopePosition = Scope(title='Position (rad)', timeWindow=10,
                              xLabel='Time (s)', yLabel='Position (rad)')
        scopePosition.attachSignal(name='position (rads)', width=1)
        scopePosition.attachSignal(name='Amplitude (rad)', color=[76, 114, 176], width=2, lineStyle='--')
        scopeVoltage = Scope(title='Vm (V)', timeWindow=10, xLabel='Time (s)', yLabel='Voltage')
        scopeVoltage.attachSignal(name='Voltage')
    log = {}
    thread = Thread(target=lambda: control_loop(ctrl, log, scopePosition, scopeVoltage))
    thread.start()
    while thread.is_alive() and not KILL_THREAD:
        if scopes:
            Scope.refreshAll()          # hay que llamarlo desde el hilo principal
        time.sleep(0.01)
    thread.join()
    return log if log.get('completa') else None


def informe(ctrl, log):
    t, r, y, u, wall = (log[k] for k in ('t', 'r', 'y', 'u', 'wall'))
    dt = np.diff(wall)
    print(f'\n--- {ctrl.descripcion}  ({len(t)} muestras, Ts medio {1e3 * dt.mean():.2f} ms, '
          f'maximo {1e3 * dt.max():.2f} ms) ---')
    if metricas_prueba is not None:
        m = metricas_prueba(t, r, y, u)
        if m:
            for nombre, valor in m.items():
                print(f'{nombre:<9}= {valor:.5f}')
    else:
        print('(pon promediar_resultados.py en esta carpeta para ver los indices al terminar)')
    os.makedirs(CARPETA, exist_ok=True)
    n = 1
    while os.path.exists(f'{CARPETA}/prueba_{ctrl.etiqueta}_run{n}.csv'):
        n += 1
    fname = f'{CARPETA}/prueba_{ctrl.etiqueta}_run{n}.csv'
    np.savetxt(fname, np.column_stack([t, r, y, u, wall]), delimiter=',',
               header='t,r,y,u,t_wall', comments='')
    print(f'Datos guardados en {fname}')


def probar(fabrica, args):
    """Repite la prueba args.rep veces con un controlador nuevo cada vez."""
    scopes = not args.sin_graficas and args.rep == 1
    if args.rep > 1 and not args.sin_graficas:
        print('Con --rep mayor que 1 las graficas en vivo se desactivan.')
    ctrl = None
    for i in range(args.rep):
        ctrl = fabrica()
        if i == 0:
            print(f'\n{ctrl.descripcion}')
        if args.rep > 1:
            print(f'Prueba {i + 1} de {args.rep}...')
        log = ejecutar(ctrl, scopes)
        if log is None:
            print('Prueba interrumpida: no se guarda.')
            return
        informe(ctrl, log)
        if i < args.rep - 1:
            time.sleep(1.0)             # deja que el disco se pare entre pruebas
    if scopes and not KILL_THREAD:
        print('\nGraficas abiertas. Pulsa Ctrl+C para cerrar.')
        while not KILL_THREAD:
            Scope.refreshAll()
            time.sleep(0.01)
#endregion


#region: Identificacion del motor
def identificar(voltaje):
    """Escalones de +V y -V en lazo abierto; ajusta w(t) = K*V*(1 - exp(-t/tau)) a cada uno."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        sys.exit('Falta scipy para identificar:  pip install scipy')
    # (inicio, fin, voltaje): reposo, +V, el disco se frena, -V, reposo
    plan = [(0.2, 1.4, +voltaje), (2.4, 3.6, -voltaje)]
    T_TOTAL = 4.6
    N = int(round(T_TOTAL / TS)) + 1
    t = np.arange(N) * TS
    u, w = np.zeros(N), np.zeros(N)
    for a, b, v in plan:
        u[(t >= a) & (t < b)] = v
    done = 0
    print(f'Identificando con escalones de +-{voltaje:g} V (el disco girara unos segundos)...')
    with QubeServo3(hardware=1, frequency=FRECUENCIA) as myQube:
        for k in range(N):
            if KILL_THREAD:
                break
            myQube.read_outputs()
            w[k] = float(myQube.motorSpeed)
            myQube.write_voltage(float(u[k]))
            done = k + 1
    if done < N:
        print('Interrumpido: no se guarda nada.')
        return

    def modelo(tt, K, tau):
        return K * (1.0 - np.exp(-tt / tau))

    resultados = []
    for a, b, v in plan:
        m = (t >= a) & (t < b)
        tt, gan = t[m] - a, w[m] / v                    # ganancia en rad/s por V (el -V da el mismo signo)
        try:
            (K, tau), _ = curve_fit(modelo, tt, gan, p0=[max(gan[-50:].mean(), 1.0), 0.1])
        except RuntimeError:
            sys.exit('No se pudo ajustar el escalon (revisa que el disco gire y que la medida de velocidad sea correcta).')
        rms = float(np.sqrt(np.mean((modelo(tt, K, tau) - gan) ** 2)))
        print(f'  escalon {v:+g} V:  K = {K:.2f} rad/s/V   tau = {1e3 * tau:.1f} ms   (error de ajuste {rms:.2f} rad/s/V)')
        resultados.append((K, tau))
    K = float(np.mean([r[0] for r in resultados]))
    tau = float(np.mean([r[1] for r in resultados]))
    if not (K > 0 and 0.005 < tau < 2.0):
        sys.exit(f'Resultado poco creible (K={K:.2f}, tau={tau:.3f}); no se guarda.')
    with open(MODELO_JSON, 'w', encoding='utf-8') as fh:
        json.dump({'K': K, 'tau': tau, 'voltaje_escalon': voltaje,
                   'fecha': time.strftime('%Y-%m-%d %H:%M:%S')}, fh, indent=2)
    print(f'\nModelo del disco: K = {K:.2f} rad/s/V, tau = {tau:.4f} s  -> guardado en {MODELO_JSON}')
    print('lqr y lqi lo leen automaticamente.')
#endregion


def main():
    ap = argparse.ArgumentParser(description='Control de posicion del disco: cascada, LQR y LQI.')
    sub = ap.add_subparsers(dest='algoritmo', required=True)
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument('--rep', type=int, default=1, help='numero de pruebas seguidas (por defecto 1)')
    comun.add_argument('--sin-graficas', action='store_true', help='sin graficas en vivo (menos jitter)')
    modelo = argparse.ArgumentParser(add_help=False)
    modelo.add_argument('--k', type=float, help='ganancia del motor K en rad/s/V (por defecto modelo_disco.json)')
    modelo.add_argument('--tau', type=float, help='constante de tiempo tau en s (por defecto modelo_disco.json)')

    s = sub.add_parser('identificar', help='mide K y tau del motor con un escalon de voltaje')
    s.add_argument('--voltaje', type=float, default=2.0, help='amplitud del escalon en V (por defecto 2)')

    s = sub.add_parser('cascada', parents=[comun], help='P de posicion + PI de velocidad')
    s.add_argument('--kp', type=float, default=15.0, help='ganancia P de posicion (rad/s por rad)')
    s.add_argument('--kpv', type=float, default=0.3, help='ganancia P del lazo de velocidad (V por rad/s)')
    s.add_argument('--kiv', type=float, default=1.0, help='ganancia I del lazo de velocidad (V por rad)')

    s = sub.add_parser('lqr', parents=[comun, modelo], help='LQR de 2 estados (posicion, velocidad)')
    s.add_argument('--qp', type=float, default=50.0, help='peso del error de posicion')
    s.add_argument('--qw', type=float, default=0.1, help='peso de la velocidad')
    s.add_argument('--r', type=float, default=1.0, help='peso del voltaje')

    s = sub.add_parser('lqi', parents=[comun, modelo], help='LQR con accion integral')
    s.add_argument('--qp', type=float, default=50.0, help='peso del error de posicion')
    s.add_argument('--qw', type=float, default=0.1, help='peso de la velocidad')
    s.add_argument('--qi', type=float, default=5.0, help='peso de la integral del error')
    s.add_argument('--r', type=float, default=1.0, help='peso del voltaje')

    args = ap.parse_args()
    if getattr(args, 'rep', 1) < 1:
        ap.error('--rep debe ser al menos 1')

    if args.algoritmo == 'identificar':
        if not 0 < args.voltaje <= 5:
            ap.error('--voltaje debe estar entre 0 y 5 V')
        identificar(args.voltaje)
        return

    if args.algoritmo == 'cascada':
        probar(lambda: Cascada(args.kp, args.kpv, args.kiv), args)
        return

    # lqr / lqi
    K, tau, medido = cargar_modelo()
    K = args.k if args.k else K
    tau = args.tau if args.tau else tau
    if not medido and not (args.k and args.tau):
        print(f'AVISO: no hay {MODELO_JSON}; se usan valores tipicos K={K:g}, tau={tau:g}. '
              f'Ejecuta primero:  py pos_cascada_lqr.py identificar')
    es_lqi = args.algoritmo == 'lqi'
    k0, k1, k2 = ganancias_lqr(K, tau, args.qp, args.qw, args.r, args.qi if es_lqi else None)
    print(f'Modelo K={K:.2f} rad/s/V, tau={tau:.4f} s')
    print(f'Ganancias LQ:  K0={k0:.4g} V/rad   K1={k1:.4g} V/(rad/s)' + (f'   K2={k2:.4g} V/(rad*s)' if es_lqi else ''))
    if not es_lqi:
        print(f'(equivale a un PD con Kp={k0:.4g}, Kd={k1:.4g})')
    etiqueta = f'{args.algoritmo}_qp{args.qp:g}_qw{args.qw:g}' + (f'_qi{args.qi:g}' if es_lqi else '') + f'_r{args.r:g}'
    desc = f'{args.algoritmo.upper()}  qp={args.qp:g}  qw={args.qw:g}' + (f'  qi={args.qi:g}' if es_lqi else '') + f'  r={args.r:g}'
    probar(lambda: EstadoLQ(k0, k1, k2, etiqueta, desc), args)


if __name__ == '__main__':
    main()
