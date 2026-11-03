## pid_control_qube_task.py
# Basado en pd_control_qube_task.py (Quanser).
# Controlador P, PI o PID seleccionable para la POSICION o la VELOCIDAD del disco
# del QUBE-Servo 3.
# Al terminar guarda un CSV (t, referencia, medida, voltaje) e imprime IAE, ISE, ITAE,
# ITSE y esfuerzo de control.
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# imports
from threading import Thread
import signal
import time
import numpy as np
from pal.products.qube import QubeServo2, QubeServo3
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
TF_D     = 0.02       # s, filtro de la derivada (solo se usa en velocidad)

# Ganancias (kp, ki, kd) para cada objetivo y modo.
# P, PI y PID se distinguen solo por tener ki y/o kd a cero.
TARGET = 'position'   # 'position' (rad) o 'speed' (rad/s)
REF_AMPLITUDE = {'position': 1.5, 'speed': 20.0}   # rad  /  rad/s
LABEL = {'position': ('Position', 'rad'), 'speed': ('Speed', 'rad/s')}

kp, ki, kd = [7.0, 0.0, 0.2]   # PID
color = np.array([0, 1, 0], dtype=np.float64)
#endregion


#region: Setup de referencia y graficas
squareWave = SignalGenerator().square(REF_AMPLITUDE[TARGET], REF_PERIOD)
next(squareWave)

name, unit = LABEL[TARGET]
scopeMain = Scope(
    title=f'Measured vs Desired {name} PID',
    timeWindow=10,
    xLabel='Time (s)',
    yLabel=f'{name} ({unit})')
scopeMain.attachSignal(name=f'Measured {name}', width=1)
scopeMain.attachSignal(name=f'Desired {name}', color=[76,114,176], width=2, lineStyle='--')

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

    def update(self, ref, meas, meas_rate, dt):
        e = ref - meas
        u_raw = self.kp*e + self.integral - self.kd*meas_rate
        u = float(np.clip(u_raw, -self.u_max, self.u_max))
        # solo integra si no esta saturado, o si el error ayuda a salir de la saturacion
        if dt > 0 and (u == u_raw or e*u_raw < 0):
            self.integral += self.ki*e*dt
        return u
#endregion


#region: Resultados
def report(log):
    d = np.array(log)
    if len(d) < 2:
        return
    t, ref, y, u, tau = d.T          # tau = tiempo desde el ultimo cambio de referencia
    e = ref - y
    dt = np.diff(t, prepend=t[0])
    metrics = {
        'IAE':      np.sum(np.abs(e)*dt),
        'ISE':      np.sum(e**2*dt),
        'ITAE':     np.sum(tau*np.abs(e)*dt),
        'ITSE':     np.sum(tau*e**2*dt),
        'Esfuerzo': np.sum(u**2*dt),
    }
    print(f'\n--- {TARGET} / PID  (kp={kp}, ki={ki}, kd={kd}) ---')
    for k, v in metrics.items():
        print(f'{k:9s} = {v:.4f}')
    fname = f'resultados_{TARGET}_PID.csv'
    np.savetxt(fname, d[:, :4], delimiter=',', header='t,ref,medida,voltaje', comments='')
    print(f'Datos guardados en {fname}')
#endregion


# Code to control the Qube Hardware
def control_loop():
    QubeClass = QubeServo3
    pid = PID(kp, ki, kd, V_MAX)
    log = []

    with QubeClass(hardware=1, pendulum=0) as myQube:
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

            # Medida y su derivada
            dt = timeStamp - prev_t
            if TARGET == 'position':
                meas = myQube.motorPosition
                meas_rate = myQube.motorSpeed            # d(posicion)/dt la da el sensor
            else:
                meas = myQube.motorSpeed
                if dt > 0:                               # derivada filtrada de la velocidad
                    raw = (meas - prev_meas) / dt
                    meas_rate += (raw - meas_rate) * dt / (TF_D + dt)

            # Controller
            voltage = pid.update(desired, meas, meas_rate, dt)

            # Write commands
            myQube.write_led(color)
            myQube.write_voltage(voltage)

            scopeMain.sample(timeStamp, [meas, desired])
            scopeVoltage.sample(timeStamp, [voltage])
            log.append((timeStamp, desired, meas, voltage, timeStamp - t_ref))

            prev_t, prev_meas = timeStamp, meas
            timeStamp = elapsed_time()

    report(log)


# Setup data generation thread and run until complete
thread = Thread(target=control_loop)
thread.start()

while thread.is_alive() and (not KILL_THREAD):
    Scope.refreshAll()
    time.sleep(0.01)

input('Presiona cualquier botón para salir.')
