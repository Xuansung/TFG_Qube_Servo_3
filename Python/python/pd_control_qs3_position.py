## pd_control_qs3_position.py
# Traduccion a Python del modelo de Simulink pd_control_qs3.slx (QUARC, Qube-Servo 3).
# Control PD de la POSICION del disco con una onda cuadrada de referencia:
#
#       u = sat_10V( Kp*e  - Kd*w )
#
# Uso:   py pd_control_qs3.py            (Kp=7, Kd=0.2)
#        py pd_control_qs3.py 5 0.15     (otros Kp y Kd)
# Cada ejecucion guarda un CSV en pruebas_python/ con t, r, y, u y el tiempo real medido.
# La diferencia respecto a los otros es que hace una media de los indices sobre todos los 
# tramos de referencia, en lugar de sumarlos sobre toda la prueba.

import os
import signal
import sys
import time
from threading import Thread
import numpy as np
from pal.products.qube import QubeServo3
from pal.utilities.math import SignalGenerator
from pal.utilities.scope import Scope

#region: Configuracion
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0     # Proportional Gain
KD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2     # Derivative Gain

AMPLITUD = 1.5        # rad   
PERIODO = 2.5         # s  
V_MAX = 10.0          # V 
FRECUENCIA = 500      # Hz
TS = 1.0 / FRECUENCIA
STOP_TIME = 9.99      # s     
SCOPES = False         # False = sin graficas en vivo (menos carga de CPU)
#endregion

#Hilo para poder matar el bucle de control con Ctrl-C
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

"Calcula metricas de control: IAE, ITAE, ISE y TV (total variation) "
"para cada tramo entre cambios de referencia consecutivos."
def metricas(t, r, y, u, Ts=TS):
    cambios = np.where(np.abs(np.diff(r)) > 0.5)[0] # find(abs(diff(r)) > 0.5)
    IAE, ITAE, ISE, TV = [], [], [], []
    for n in range(1, len(cambios)):
        a = cambios[n - 1] + 1                      # primer dato tras el cambio anterior
        b = cambios[n]                              # ultimo dato antes del siguiente cambio
        e = r[a:b + 1] - y[a:b + 1]                 # error acumulado en el tramo
        tseg = t[a:b + 1] - t[a]                    # tiempo relativo al inicio del tramo
        IAE.append(np.sum(np.abs(e)) * Ts)          # IAE
        ITAE.append(np.sum(tseg * np.abs(e)) * Ts)  # ITAE
        ISE.append(np.sum(e**2) * Ts)               # ISE
        TV.append(np.sum(np.abs(np.diff(u[a:b + 1]))))  #TV
    return dict(IAE=np.mean(IAE), ITAE=np.mean(ITAE), ISE=np.mean(ISE), #media
                TV=np.mean(TV), tramos=len(IAE))

def control_loop(log, scopePosition, scopeVoltage):
    squareWave = SignalGenerator().square(AMPLITUD, PERIODO)    # Smooth Signal Generator x Gain
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
            myQube.read_outputs()                  # HIL Read Timebase: espera a la siguiente muestra
            wall[k] = time.perf_counter() - start
            r[k] = squareWave.send(t[k])           # referencia 
            position = float(myQube.motorPosition)    # rad
            speed = float(myQube.motorSpeed)       # rad/s
            u[k] = np.clip(KP * (r[k] - position) - KD * speed, -V_MAX, V_MAX)
            myQube.write_voltage(u[k])             # HIL Write
            y[k] = position
            done = k + 1
            if scopePosition is not None:
                scopePosition.sample(t[k], [position, r[k]])
                scopeVoltage.sample(t[k], [u[k]])
    log.update(t=t[:done], r=r[:done], y=y[:done], u=u[:done], wall=wall[:done])

def informe(log):
    t, r, y, u, wall = (log[k] for k in ('t', 'r', 'y', 'u', 'wall'))
    if len(t) < 3:
        print('Sin datos suficientes.')
        return
    m = metricas(t, r, y, u)
    print(f'\n--- Kp={KP:g}  Kd={KD:g}  ({len(t)} muestras, {m["tramos"]} tramos) ---')
    print(f'IAE  = {m["IAE"]:.5f}')
    print(f'ITAE = {m["ITAE"]:.5f}')
    print(f'ISE  = {m["ISE"]:.5f}')
    print(f'TV   = {m["TV"]:.5f}')
    os.makedirs('pruebas_python', exist_ok=True)
    n = 1
    while os.path.exists(f'pruebas_python/posicion_kp{KP:g}_kd{KD:g}_run{n}.csv'):
        n += 1
    fname = f'pruebas_python/prueba_kp{KP:g}_kd{KD:g}_run{n}.csv'
    np.savetxt(fname, np.column_stack([t, r, y, u, wall]), delimiter=',',
               header='t,r,y,u,t_wall', comments='')
    print(f'Datos guardados en {fname}')

if __name__ == '__main__':
    scopePosition = scopeVoltage = None
    if SCOPES:
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
    if SCOPES and not KILL_THREAD:
        print('Prueba terminada. Ctrl+C para cerrar las graficas.')
        while not KILL_THREAD:
            Scope.refreshAll()
            time.sleep(0.01)