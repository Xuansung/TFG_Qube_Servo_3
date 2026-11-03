## promediar_resultados.py
# Promedia las pruebas guardadas por los scripts de Python (posicion o velocidad),
# agrupa las que tienen el mismo algoritmo y las mismas ganancias y dibuja un ranking
# de mejor a peor.
#
# Uso:   py promediar_resultados.py posicion ITAE
#        py promediar_resultados.py posicion ISE --mejores            (la mejor de cada algoritmo)
#        py promediar_resultados.py posicion ITAE --algoritmo PD,CASCADA,LQI
#        py promediar_resultados.py velocidad ISE --top 10 --guardar
#
# Indices: IAE, ITAE, ISE, TV, Esfuerzo (en todos, menor = mejor).
#
# Nombres de CSV que entiende (carpeta pruebas_python/):
#   prueba_kp7_kd0.2_run1.csv                     posicion, PD         (pd_control_qs3_position.py)
#   prueba_cascada_kp20_kpv0.3_kiv1_run1.csv      posicion, cascada    (pos_cascada_lqr.py)
#   prueba_lqr_qp50_qw0.1_r1_run1.csv             posicion, LQR        (pos_cascada_lqr.py)
#   prueba_lqi_qp50_qw0.1_qi5_r1_run1.csv         posicion, LQR + integral
#   vel_kp0.1_ki5_kd0.002_run1.csv                velocidad; P, PI o PID segun ki y kd
# Si el nombre no lleva algoritmo se deduce (posicion = PD; velocidad = P, PI o PID).
# Cualquier algoritmo nuevo vale: basta con poner su nombre justo despues del prefijo
# y despues las ganancias como <nombre><valor> separadas por guiones bajos.
#
# Los indices se calculan igual que en la bitacora: por tramo (entre dos cambios de
# referencia), sin el primero ni el ultimo, y promediando los tramos de cada prueba.
# Despues se promedian las pruebas que comparten algoritmo y ganancias (media y desviacion).

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INDICES = ['IAE', 'ITAE', 'ISE', 'TV', 'Esfuerzo']

MODOS = {
    'posicion':  {'prefijo': 'prueba',
                  'unidades': {'IAE': 'rad·s', 'ITAE': 'rad·s²', 'ISE': 'rad²·s',
                               'TV': 'V', 'Esfuerzo': 'V²·s'}},
    'velocidad': {'prefijo': 'vel',
                  'unidades': {'IAE': 'rad', 'ITAE': 'rad·s', 'ISE': 'rad²/s',
                               'TV': 'V', 'Esfuerzo': 'V²·s'}},
}

FICHERO = re.compile(r'^(?P<prefijo>[a-z]+)_(?P<resto>.+)_run(?P<run>\d+)$')
TOKEN = re.compile(r'^(?P<nombre>[A-Za-z]+)(?P<valor>-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)$')


#region: Lectura y calculo de indices
def leer_ganancias(stem, prefijo):
    """'prueba_cascada_kp20_kpv0.3_run1' -> ('CASCADA', (('Kp', 20.0), ('Kpv', 0.3))).
    Devuelve None si el nombre no encaja."""
    m = FICHERO.match(stem)
    if not m or m['prefijo'] != prefijo:
        return None
    tokens = m['resto'].split('_')
    algoritmo = None
    if not TOKEN.match(tokens[0]):              # primer token sin numero = nombre del algoritmo
        algoritmo = tokens.pop(0).upper()
    params = []
    for tk in tokens:
        t = TOKEN.match(tk)
        if not t:
            return None
        params.append((t['nombre'].capitalize(), float(t['valor'])))
    if not params:
        return None
    if algoritmo is None:                       # nombres antiguos, sin algoritmo
        g = dict(params)
        if prefijo == 'vel':
            algoritmo = 'PID' if g.get('Kd') else ('PI' if g.get('Ki') else 'P')
        else:
            algoritmo = 'PD'
    return algoritmo, tuple(params)


def leer_csv(ruta):
    """Todas las familias de CSV guardan t, referencia, medida y voltaje en las 4 primeras columnas."""
    d = np.loadtxt(ruta, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3), ndmin=2)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def metricas_prueba(t, r, y, u):
    """Indices de una prueba: media de los tramos entre cambios de referencia consecutivos."""
    if len(t) < 3:
        return None
    dt = np.append(np.diff(t), np.median(np.diff(t)))     # sirve con Ts fijo o variable
    umbral = 0.25 * (r.max() - r.min())                   # salto de la onda cuadrada
    cambios = np.where(np.abs(np.diff(r)) > umbral)[0]
    res = {k: [] for k in INDICES}
    for n in range(1, len(cambios)):
        a = cambios[n - 1] + 1                            # primer dato tras el cambio anterior
        b = cambios[n]                                    # ultimo dato antes del siguiente
        e = r[a:b + 1] - y[a:b + 1]
        tseg = t[a:b + 1] - t[a]
        d = dt[a:b + 1]
        uu = u[a:b + 1]
        res['IAE'].append(np.sum(np.abs(e) * d))
        res['ITAE'].append(np.sum(tseg * np.abs(e) * d))
        res['ISE'].append(np.sum(e**2 * d))
        res['TV'].append(np.sum(np.abs(np.diff(uu))))
        res['Esfuerzo'].append(np.sum(uu**2 * d))
    if not res['IAE']:
        return None
    return {k: float(np.mean(v)) for k, v in res.items()}


def cargar(carpeta, modo, algoritmos=None):
    """Devuelve {(algoritmo, ganancias): [indices de cada prueba]}."""
    prefijo = MODOS[modo]['prefijo']
    grupos = defaultdict(list)
    for ruta in sorted(carpeta.glob(f'{prefijo}_*_run*.csv')):
        clave = leer_ganancias(ruta.stem, prefijo)
        if clave is None:
            continue
        if algoritmos and clave[0] not in algoritmos:
            continue
        try:
            m = metricas_prueba(*leer_csv(ruta))
        except ValueError:
            print(f'Aviso: {ruta.name} no se pudo leer, se ignora.')
            continue
        if m is None:
            print(f'Aviso: {ruta.name} no tiene tramos completos, se ignora.')
            continue
        grupos[clave].append(m)
    return grupos


def resumir(grupos, indice):
    """Media y desviacion de cada configuracion, ordenadas por el indice (y por ITAE si empatan)."""
    filas = []
    for (algoritmo, params), runs in grupos.items():
        M = np.array([[m[k] for k in INDICES] for m in runs])
        std = M.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(INDICES))
        filas.append({'algoritmo': algoritmo, 'params': params, 'n': len(runs),
                      'media': dict(zip(INDICES, M.mean(axis=0))),
                      'std': dict(zip(INDICES, std))})
    filas.sort(key=lambda f: (f['media'][indice], f['media']['ITAE']))
    return filas


def mejores_por_algoritmo(filas):
    """Se queda con la primera (la mejor) configuracion de cada algoritmo."""
    vistos, mejores = set(), []
    for f in filas:
        if f['algoritmo'] not in vistos:
            vistos.add(f['algoritmo'])
            mejores.append(f)
    return mejores
#endregion


#region: Salida
def etiqueta(fila):
    return f'{fila["algoritmo"]}  ' + '  '.join(f'{n}={v:g}' for n, v in fila['params'])


def imprimir_tabla(filas, indice):
    print(f'\nOrdenado por {indice} (menor = mejor)\n')
    ancho = max(len('Algoritmo y ganancias'), max(len(etiqueta(f)) for f in filas))
    cab = [f'[{k}]' if k == indice else k for k in INDICES]
    print(f'{"#":>3}  {"Algoritmo y ganancias":<{ancho}} {"n":>2}  ' + '  '.join(f'{c:>18}' for c in cab))
    for i, f in enumerate(filas, 1):
        celdas = '  '.join(f'{f["media"][k]:.5g} ± {f["std"][k]:.2g}'.rjust(18) for k in INDICES)
        print(f'{i:>3}  {etiqueta(f):<{ancho}} {f["n"]:>2}  {celdas}')
    if any(f['n'] == 1 for f in filas):
        print('\nNota: con n=1 no hay desviacion (se muestra 0). Repite la prueba para tenerla.')


def guardar_csv(filas, ruta):
    nombres = []
    for f in filas:
        for n, _ in f['params']:
            if n not in nombres:
                nombres.append(n)
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['algoritmo'] + nombres + ['n']
                   + [f'{k}_media' for k in INDICES] + [f'{k}_std' for k in INDICES])
        for f in filas:
            g = dict(f['params'])
            w.writerow([f['algoritmo']] + [g.get(p, '') for p in nombres] + [f['n']]
                       + [f['media'][k] for k in INDICES] + [f['std'][k] for k in INDICES])


def graficar(filas, indice, modo, top, log, destino, extra=''):
    sel = filas[:top]
    unidad = MODOS[modo]['unidades'][indice]
    valores = np.array([f['media'][indice] for f in sel])
    sigma = np.array([f['std'][indice] for f in sel])
    inferior = np.minimum(sigma, 0.99 * valores) if log else sigma
    colores = plt.get_cmap('RdYlGn_r')(np.linspace(0.05, 0.95, len(sel)))   # verde = mejor

    fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.45 * len(sel) + 1.8)))
    pos = np.arange(len(sel))
    ax.barh(pos, valores, xerr=[inferior, sigma], color=colores,
            edgecolor='black', linewidth=0.5, capsize=3)
    for i, (v, s) in enumerate(zip(valores, sigma)):
        ax.text(v + s, i, f' {v:.4g}', va='center', fontsize=8)
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{etiqueta(f)}  (n={f["n"]})' for f in sel])
    ax.invert_yaxis()                                   # el mejor arriba
    if log:
        ax.set_xscale('log')
    ax.margins(x=0.15)
    ax.set_xlabel(f'{indice} medio ± desviacion ({unidad})')
    ax.set_title(f'{modo.capitalize()}: ordenado por {indice} (menor = mejor){extra}'
                 + (f'  [{len(sel)} de {len(filas)}]' if len(sel) < len(filas) else ''))
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    if destino:
        fig.savefig(destino, dpi=150, bbox_inches='tight')
        print(f'Grafica guardada en {destino}')
    plt.show()
#endregion


def main():
    def normalizar_modo(s):
        return s.lower().replace('ó', 'o')

    def normalizar_indice(s):
        return next((k for k in INDICES if k.lower() == s.lower()), s)

    ap = argparse.ArgumentParser(description='Promedia y ordena las pruebas del Qube-Servo 3.')
    ap.add_argument('modo', type=normalizar_modo, choices=list(MODOS),
                    help='posicion o velocidad')
    ap.add_argument('indice', type=normalizar_indice, choices=INDICES,
                    help='indice por el que se ordena: ' + ', '.join(INDICES))
    ap.add_argument('--carpeta', default='pruebas_python',
                    help='carpeta con los CSV (por defecto pruebas_python)')
    ap.add_argument('--algoritmo', default='',
                    help='solo estos algoritmos, separados por comas (p. ej. PD,CASCADA,LQI)')
    ap.add_argument('--mejores', action='store_true',
                    help='muestra solo la mejor configuracion de cada algoritmo')
    ap.add_argument('--top', type=int, default=15,
                    help='cuantas configuraciones dibujar, las mejores (por defecto 15)')
    ap.add_argument('--log', action='store_true', help='eje x logaritmico')
    ap.add_argument('--guardar', action='store_true',
                    help='guarda la grafica (PNG) y la tabla (CSV) en la carpeta')
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    algoritmos = {a.strip().upper() for a in args.algoritmo.split(',') if a.strip()}
    grupos = cargar(carpeta, args.modo, algoritmos)
    if not grupos:
        ap.exit(1, f'No hay CSV de {args.modo} en {carpeta}/ '
                   f'(se buscan {MODOS[args.modo]["prefijo"]}_*_run*.csv'
                   + (f', algoritmos {sorted(algoritmos)}' if algoritmos else '') + ').\n')

    filas = resumir(grupos, args.indice)
    n_alg = len({f['algoritmo'] for f in filas})
    print(f'{sum(f["n"] for f in filas)} pruebas, {len(filas)} configuraciones, '
          f'{n_alg} algoritmo{"s" if n_alg != 1 else ""}.')
    extra, sufijo = '', ''
    if args.mejores:
        filas = mejores_por_algoritmo(filas)
        extra, sufijo = ' · mejor de cada algoritmo', '_mejores'
    imprimir_tabla(filas, args.indice)

    destino = None
    if args.guardar:
        tabla = carpeta / f'resumen_{args.modo}_{args.indice}{sufijo}.csv'
        guardar_csv(filas, tabla)
        print(f'Tabla guardada en {tabla}')
        destino = carpeta / f'ranking_{args.modo}_{args.indice}{sufijo}.png'
    graficar(filas, args.indice, args.modo, args.top, args.log, destino, extra)


if __name__ == '__main__':
    main()
