## pd_control_qs3.py
# Traduccion a Python del modelo de Simulink pd_control_qs3.slx (QUARC, Qube-Servo 3).
# Control PD de la POSICION del disco con una onda cuadrada de referencia:
#
#       u = sat_10V( Kp*(r - theta) - Kd*omega )
#
# Correspondencia bloque a bloque con el .slx:
#   Smooth Signal Generator (Square, amp 1, 0.4 Hz) x Gain 'Amplitude (rad)' = 1.5
#                           -> SignalGenerator().square(AMPLITUD, PERIODO)   (+-1.5 rad, periodo 2.5 s)
#   Qube-Servo 3 - IO: discEncoder (counts) x 'counts to rads' (2*pi/512/4)  -> myQube.motorPosition (rad)
#                      discTach (counts/s)  x 'counts/s to rad/s'            -> myQube.motorSpeed (rad/s)
#   Sum  (+ -)         -> e = r - theta
#   Proportional Gain  -> KP * e                    (Slider Gain, valor 7.0 en el modelo)
#   Derivative Gain    -> KD * omega                (Slider Gain, valor 0.2 en el modelo)
#   Sum1 (+ -)         -> u = KP*e - KD*omega       (la derivada va sobre la velocidad medida)
#   +/- 10V Limit      -> np.clip(u, -V_MAX, V_MAX)  (pal solo satura a +-15 V, por eso se hace aqui)
#   Paso fijo 0.002 s, StopTime 9.99 s -> frequency=500 Hz, 4996 muestras
#   HIL Initialize     -> QubeServo3(): mismas board_options, encoder a cero al empezar
#
# Uso:   py pd_control_qs3.py            (Kp=7, Kd=0.2, los valores del modelo)
#        py pd_control_qs3.py 5 0.15     (otros Kp y Kd)
# Cada ejecucion guarda un CSV en pruebas_python/ con t, r, y, u y el tiempo real medido.

import os
import signal
import sys
import time
from threading import Thread

import numpy as np

# ---------------------------------------------------------------------------
#region: Configuracion (valores del modelo Simulink)
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0     # Proportional Gain
KD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2     # Derivative Gain
AMPLITUD = 1.5        # rad   -> 'Amplitude (rad)' x amplitud 1 del generador
PERIODO = 2.5         # s     -> 0.4 Hz. OJO: en pal, square(amplitud, PERIODO), no frecuencia
V_MAX = 10.0          # V     -> '+/- 10V Limit'
FRECUENCIA = 500      # Hz    -> paso fijo de 0.002 s
TS = 1.0 / FRECUENCIA
STOP_TIME = 9.99      # s     -> 4996 muestras, como en Simulink
HARDWARE = 1          # 1 = Qube fisico (0 = virtual, necesita Quanser Interactive Labs)
SCOPES = True         # False = sin graficas en vivo (menos carga de CPU, menos jitter)
#endregion
# ---------------------------------------------------------------------------

KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)


def metricas(t, r, y, u, Ts=TS):
    """Misma definicion que analizar_pruebas.m: media de IAE, ITAE, sobrepaso (%) y TV
    sobre cada tramo entre dos cambios de referencia consecutivos."""
    cambios = np.where(np.abs(np.diff(r)) > 0.5)[0]        # find(abs(diff(r)) > 0.5)
    IAE, ITAE, SOBREPASO, TV = [], [], [], []
    for n in range(1, len(cambios)):
        a = cambios[n - 1] + 1                              # primer dato tras el cambio anterior
        b = cambios[n]                                      # ultimo dato antes del siguiente cambio
        e = r[a:b + 1] - y[a:b + 1]
        tseg = t[a:b + 1] - t[a]
        IAE.append(np.sum(np.abs(e)) * Ts)
        ITAE.append(np.sum(tseg * np.abs(e)) * Ts)
        salto = r[a] - y[a]
        yn = (y[a:b + 1] - y[a]) / salto
        SOBREPASO.append(max(0.0, np.max(yn) - 1) * 100)
        TV.append(np.sum(np.abs(np.diff(u[a:b + 1]))))
    return dict(IAE=np.mean(IAE), ITAE=np.mean(ITAE), SOBREPASO=np.mean(SOBREPASO),
                TV=np.mean(TV), tramos=len(IAE))


def control_loop(log, scopePosition, scopeVoltage):
    from pal.products.qube import QubeServo3
    from pal.utilities.math import SignalGenerator

    squareWave = SignalGenerator().square(AMPLITUD, PERIODO)    # Smooth Signal Generator x Gain
    next(squareWave)

    N = int(round(STOP_TIME / TS)) + 1
    t = np.arange(N) * TS
    r, y, u, wall = (np.zeros(N) for _ in range(4))
    done = 0

    # QubeServo3 usa por defecto 500 Hz y las mismas board_options que el HIL Initialize del .slx
    with QubeServo3(hardware=HARDWARE, frequency=FRECUENCIA) as myQube:
        start = time.perf_counter()
        for k in range(N):
            if KILL_THREAD:
                break

            myQube.read_outputs()                  # HIL Read Timebase: espera a la siguiente muestra
            wall[k] = time.perf_counter() - start

            r[k] = squareWave.send(t[k])           # referencia (tiempo = k*Ts, como el paso fijo de Simulink)
            theta = float(myQube.motorPosition)    # rad
            omega = float(myQube.motorSpeed)       # rad/s

            u[k] = np.clip(KP * (r[k] - theta) - KD * omega, -V_MAX, V_MAX)
            myQube.write_voltage(u[k])             # HIL Write
            y[k] = theta
            done = k + 1

            if scopePosition is not None:
                scopePosition.sample(t[k], [theta, r[k]])
                scopeVoltage.sample(t[k], [u[k]])

    log.update(t=t[:done], r=r[:done], y=y[:done], u=u[:done], wall=wall[:done])


def informe(log):
    t, r, y, u, wall = (log[k] for k in ('t', 'r', 'y', 'u', 'wall'))
    if len(t) < 3:
        print('Sin datos suficientes.')
        return
    m = metricas(t, r, y, u)
    dt = np.diff(wall)
    print(f'\n--- Kp={KP:g}  Kd={KD:g}  ({len(t)} muestras, {m["tramos"]} tramos) ---')
    print(f'IAE       = {m["IAE"]:.5f}')
    print(f'ITAE      = {m["ITAE"]:.5f}')
    print(f'SOBREPASO = {m["SOBREPASO"]:.3f} %')
    print(f'TV        = {m["TV"]:.4f}')
    print(f'Muestreo real: dt medio = {dt.mean()*1e3:.3f} ms, maximo = {dt.max()*1e3:.3f} ms, '
          f'{int(np.sum(dt > 3e-3))} muestras con dt > 3 ms (en Simulink: 2.000 ms fijo)')

    os.makedirs('pruebas_python', exist_ok=True)
    n = 1
    while os.path.exists(f'pruebas_python/prueba_kp{KP:g}_kd{KD:g}_run{n}.csv'):
        n += 1
    fname = f'pruebas_python/prueba_kp{KP:g}_kd{KD:g}_run{n}.csv'
    np.savetxt(fname, np.column_stack([t, r, y, u, wall]), delimiter=',',
               header='t,r,y,u,t_wall', comments='')
    print(f'Datos guardados en {fname}')


if __name__ == '__main__':
    scopePosition = scopeVoltage = None
    if SCOPES:
        from pal.utilities.scope import Scope
        scopePosition = Scope(title='Position (rad)', timeWindow=10,
                              xLabel='Time (s)', yLabel='Position (rad)')
        scopePosition.attachSignal(name='position (rads)', width=1)
        scopePosition.attachSignal(name='Amplitude (rad)', color=[76, 114, 176], width=2, lineStyle='--')
        scopeVoltage = Scope(title='Vm (V)', timeWindow=10, xLabel='Time (s)', yLabel='Voltage')
        scopeVoltage.attachSignal(name='Voltage')

    log = {}
    thread = Thread(target=lambda: control_loop(log, scopePosition, scopeVoltage))
    thread.start()
    while thread.is_alive() and not KILL_THREAD:
        if SCOPES:
            Scope.refreshAll()          # hay que llamarlo desde el hilo principal
        time.sleep(0.01)
    thread.join()
    informe(log)
