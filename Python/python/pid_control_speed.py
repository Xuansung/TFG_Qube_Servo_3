# imports
import argparse
import os
from threading import Thread
import signal
import time
import numpy as np
from pal.products.qube import QubeServo3
from pal.utilities.math import SignalGenerator
from pal.utilities.scope import Scope
 
# Setup to enable killing the data generation thread using keyboard interrupts
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
 
#region: Configuración del experimento, solo modificar esta parte.
simulationTime = 9.99   # s
V_MAX    = 10.0       # saturacion del voltaje al motor (V)
REF_PERIOD = 2.5      # s, PERIODO de la onda cuadrada
REF_AMPLITUDE = 20.0  # rad/s, la referencia salta entre +20 y -20 rad/s
TF_D     = 0.02       # s, filtro de la derivada de la velocidad (solo cuenta si kd > 0)
PAUSA    = 5.0        # s de descanso entre repeticiones (--rep)
 
# P, PI y PID se distinguen solo por tener ki y/o kd a cero.
kp, ki, kd = [0.4, 0.0, 0.00]   # PID 
color = np.array([0, 1, 0], dtype=np.float64)
#endregion

#region: Setup de referencia y graficas
squareWave = SignalGenerator().square(REF_AMPLITUDE, REF_PERIOD)
next(squareWave)
scopeMain = Scope(
    title=f'Measured vs Desired Speed PID',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Speed (rad/s)')
scopeMain.attachSignal(name='Measured Speed', width=1)
scopeMain.attachSignal(name='Desired Speed', color=[76,114,176], width=2, lineStyle='--')
scopeVoltage = Scope(
    title='Commanded Voltage',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel='Voltage')
scopeVoltage.attachSignal(name='Voltage')
#endregion
 
#region: Controlador
class PID:
    """PID con derivada sobre la medida (sin 'patada' al cambiar la referencia)
    y anti-windup por limitacion condicional de la integral."""
    def __init__(self, kp, ki, kd, u_max):
        self.kp, self.ki, self.kd, self.u_max = kp, ki, kd, u_max
        self.integral = 0.0          # estado interno: se conserva entre iteraciones
 
    def update(self, r, y, y_rate, dt):
        e = r - y
        u_raw = self.kp*e + self.integral - self.kd*y_rate
        u = float(np.clip(u_raw, -self.u_max, self.u_max))
        # solo integra si no esta saturado, o si el error ayuda a salir de la saturacion
        if dt > 0 and (u == u_raw or e*u_raw < 0):
            self.integral += self.ki*e*dt
        return u
#endregion
 
#region: Resultados
NOMBRES_INDICES = ['IAE', 'ISE', 'ITAE', 'Esfuerzo']
def indices(log):
    """IAE, ISE, ITAE y esfuerzo de una prueba (sumas sobre toda la prueba)."""
    d = np.array(log)
    t, r, y, u, tau = d.T          # tau = tiempo desde el ultimo cambio de referencia
    e = r - y
    dt = np.diff(t, prepend=t[0])
    return np.array([np.sum(np.abs(e)*dt),
                     np.sum(e**2*dt),
                     np.sum(tau*np.abs(e)*dt),
                     np.sum(u**2*dt)])
 
def guardar(log):
    """Guarda la prueba como pruebas_python/vel_kp<kp>_ki<ki>_kd<kd>_run<n>.csv sin pisar las anteriores."""
    os.makedirs('pruebas_python', exist_ok=True)
    base = f'pruebas_python/vel_kp{kp:g}_ki{ki:g}_kd{kd:g}'
    n = 1
    while os.path.exists(f'{base}_run{n}.csv'):
        n += 1
    fname = f'{base}_run{n}.csv'
    np.savetxt(fname, np.array(log)[:, :4], delimiter=',', header='t,ref,medida,voltaje', comments='')
    return fname
#endregion
 
# Code to control the Qube Hardware (una prueba completa; t_offset solo desplaza el tiempo de las graficas)
def control_loop(log, t_offset=0.0):
    pid = PID(kp, ki, kd, V_MAX)
 
    with QubeServo3(hardware=1, pendulum=0) as myQube:
        startTime = 0
        timeStamp = 0
        def elapsed_time():
            return time.time() - startTime
        startTime = time.time() 
        prev_t, prev_meas, meas_rate = 0.0, 0.0, 0.0
        prev_ref, t_ref = None, 0.0
 
        while timeStamp < simulationTime and not KILL_THREAD:
            # Read sensor information
            myQube.read_outputs()
            # Command square wave
            desired = squareWave.send(timeStamp)
            if desired != prev_ref:          # nuevo escalon: reinicia el tiempo para ITAE/ITSE
                prev_ref, t_ref = desired, timeStamp
            # Medida (velocidad) y su derivada filtrada
            dt = timeStamp - prev_t
            meas = myQube.motorSpeed
            if dt > 0:
                raw = (meas - prev_meas) / dt
                meas_rate += (raw - meas_rate) * dt / (TF_D + dt)
            # Controller
            voltage = pid.update(desired, meas, meas_rate, dt)
            # Write commands
            myQube.write_led(color)
            myQube.write_voltage(voltage)
            scopeMain.sample(t_offset + timeStamp, [meas, desired])
            scopeVoltage.sample(t_offset + timeStamp, [voltage])
            log.append((timeStamp, desired, meas, voltage, timeStamp - t_ref))
            prev_t, prev_meas = timeStamp, meas
            timeStamp = elapsed_time()
 
def esperar(segundos):
    """Pausa sin congelar las ventanas de las graficas."""
    fin = time.time() + segundos
    while time.time() < fin and not KILL_THREAD:
        Scope.refreshAll()
        time.sleep(0.01)
 
def main():
    ap = argparse.ArgumentParser(description='Control P/PI/PID de la velocidad del disco del Qube-Servo 3.')
    ap.add_argument('--rep', type=int, default=1, help='numero de pruebas seguidas con las mismas ganancias (por defecto 1)')
    args = ap.parse_args()
    resultados = []
    t_offset = 0.0                           # el tiempo de las graficas sigue creciendo entre repeticiones
    for rep in range(args.rep):
        if KILL_THREAD:
            break
        print(f'\n=== PID: kp={kp:g}, ki={ki:g}, kd={kd:g}  (repeticion {rep + 1}/{args.rep}) ===')
        log = []
        # Setup data generation thread and run until complete
        thread = Thread(target=control_loop, args=(log, t_offset))
        thread.start()
        while thread.is_alive() and (not KILL_THREAD):
            # This must be called regularly or the scope windows will freeze
            # Must be called in the main thread.
            Scope.refreshAll()
            time.sleep(0.01)
        thread.join()
        t_offset += simulationTime + PAUSA
        if len(log) < 2:
            continue
        m = indices(log)
        resultados.append(m)
        for nombre, v in zip(NOMBRES_INDICES, m):
            print(f'{nombre:9s} = {v:.4f}')
        print(f'Datos guardados en {guardar(log)}')
 
        if rep < args.rep - 1:
            esperar(PAUSA)
 
    if len(resultados) > 1:
        R = np.array(resultados)
        print(f'\n--- Resumen de {len(R)} repeticiones (PID: kp={kp:g}, ki={ki:g}, kd={kd:g}) ---')
        for i, nombre in enumerate(NOMBRES_INDICES):
            print(f'{nombre:9s} = {R[:, i].mean():.4f} ± {R[:, i].std(ddof=1):.4f}')
 
    input('Presiona cualquier botón para salir.')
 
if __name__ == '__main__':
    main()
 