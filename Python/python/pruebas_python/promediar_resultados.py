## promediar_resultados.py
# Promedia las pruebas guardadas por los scripts de Python (posicion o velocidad),
# agrupa las que tienen las mismas ganancias y dibuja un ranking de mejor a peor.
#
# Uso:   py promediar_resultados.py posicion ITAE
#        py promediar_resultados.py velocidad ISE --top 10
#        py promediar_resultados.py posicion TV --log --guardar
#
# Indices: IAE, ITAE, ISE, TV, Esfuerzo (en todos, menor = mejor).
#
# Lee los CSV de la carpeta pruebas_python/:
#   posicion:  prueba_kp<Kp>_kd<Kd>_run<n>.csv          (pd_control_qs3_position.py)
#   velocidad: vel_kp<Kp>_ki<Ki>_kd<Kd>_run<n>.csv      (pid_control_speed.py)
#
# Los indices se calculan igual que en la bitacora: por tramo (entre dos cambios de
# referencia), sin el primero ni el ultimo, y promediando los tramos de cada prueba.
# Despues se promedian las pruebas que comparten ganancias (media y desviacion).

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

FICHERO = re.compile(r'^(?P<prefijo>[a-z]+)_(?P<params>.+)_run(?P<run>\d+)$')
PARAM = re.compile(r'k([pid])(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)')


#region: Lectura y calculo de indices
def leer_ganancias(stem, prefijo):
    """'prueba_kp7_kd0.2_run1' -> (('Kp', 7.0), ('Kd', 0.2)). None si no encaja."""
    m = FICHERO.match(stem)
    if not m or m['prefijo'] != prefijo:
        return None
    pares = PARAM.findall(m['params'])
    if not pares:
        return None
    return tuple((f'K{c}', float(v)) for c, v in pares)


def leer_csv(ruta):
    """Las dos familias de CSV guardan t, referencia, medida y voltaje en las 4 primeras columnas."""
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


def cargar(carpeta, modo):
    """Devuelve {ganancias: [indices de cada prueba]}."""
    prefijo = MODOS[modo]['prefijo']
    grupos = defaultdict(list)
    for ruta in sorted(carpeta.glob(f'{prefijo}_*_run*.csv')):
        clave = leer_ganancias(ruta.stem, prefijo)
        if clave is None:
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
    for clave, runs in grupos.items():
        M = np.array([[m[k] for k in INDICES] for m in runs])
        std = M.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(INDICES))
        filas.append({'clave': clave, 'n': len(runs),
                      'media': dict(zip(INDICES, M.mean(axis=0))),
                      'std': dict(zip(INDICES, std))})
    filas.sort(key=lambda f: (f['media'][indice], f['media']['ITAE']))
    return filas
#endregion


#region: Salida
def etiqueta(clave):
    return '  '.join(f'{nombre}={valor:g}' for nombre, valor in clave)


def imprimir_tabla(filas, indice):
    print(f'\nOrdenado por {indice} (menor = mejor)\n')
    cab = [f'[{k}]' if k == indice else k for k in INDICES]
    print(f'{"#":>3}  {"Ganancias":<26} {"n":>2}  ' + '  '.join(f'{c:>18}' for c in cab))
    for i, f in enumerate(filas, 1):
        celdas = '  '.join(f'{f["media"][k]:.5g} ± {f["std"][k]:.2g}'.rjust(18) for k in INDICES)
        print(f'{i:>3}  {etiqueta(f["clave"]):<26} {f["n"]:>2}  {celdas}')
    if any(f['n'] == 1 for f in filas):
        print('\nNota: con n=1 no hay desviacion (se muestra 0). Repite la prueba para tenerla.')


def guardar_csv(filas, ruta):
    orden = ['Kp', 'Ki', 'Kd']
    params = [p for p in orden if any(p in dict(f['clave']) for f in filas)]
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(params + ['n'] + [f'{k}_media' for k in INDICES] + [f'{k}_std' for k in INDICES])
        for f in filas:
            g = dict(f['clave'])
            w.writerow([g.get(p, '') for p in params] + [f['n']]
                       + [f['media'][k] for k in INDICES] + [f['std'][k] for k in INDICES])


def graficar(filas, indice, modo, top, log, destino):
    sel = filas[:top]
    unidad = MODOS[modo]['unidades'][indice]
    valores = np.array([f['media'][indice] for f in sel])
    sigma = np.array([f['std'][indice] for f in sel])
    inferior = np.minimum(sigma, 0.99 * valores) if log else sigma
    colores = plt.get_cmap('RdYlGn_r')(np.linspace(0.05, 0.95, len(sel)))   # verde = mejor

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(sel) + 1.8)))
    pos = np.arange(len(sel))
    ax.barh(pos, valores, xerr=[inferior, sigma], color=colores,
            edgecolor='black', linewidth=0.5, capsize=3)
    for i, (v, s) in enumerate(zip(valores, sigma)):
        ax.text(v + s, i, f' {v:.4g}', va='center', fontsize=8)
    ax.set_yticks(pos)
    ax.set_yticklabels([f'{etiqueta(f["clave"])}  (n={f["n"]})' for f in sel])
    ax.invert_yaxis()                                   # el mejor arriba
    if log:
        ax.set_xscale('log')
    ax.margins(x=0.15)
    ax.set_xlabel(f'{indice} medio ± desviacion ({unidad})')
    ax.set_title(f'{modo.capitalize()}: ordenado por {indice} (menor = mejor)'
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
    ap.add_argument('--top', type=int, default=15,
                    help='cuantas configuraciones dibujar, las mejores (por defecto 15)')
    ap.add_argument('--log', action='store_true', help='eje x logaritmico')
    ap.add_argument('--guardar', action='store_true',
                    help='guarda la grafica (PNG) y la tabla (CSV) en la carpeta')
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    grupos = cargar(carpeta, args.modo)
    if not grupos:
        ap.exit(1, f'No hay CSV de {args.modo} en {carpeta}/ '
                   f'(se buscan {MODOS[args.modo]["prefijo"]}_*_run*.csv).\n')

    filas = resumir(grupos, args.indice)
    print(f'{sum(f["n"] for f in filas)} pruebas, {len(filas)} configuraciones.')
    imprimir_tabla(filas, args.indice)

    destino = None
    if args.guardar:
        guardar_csv(filas, carpeta / f'resumen_{args.modo}_{args.indice}.csv')
        print(f'Tabla guardada en {carpeta / f"resumen_{args.modo}_{args.indice}.csv"}')
        destino = carpeta / f'ranking_{args.modo}_{args.indice}.png'
    graficar(filas, args.indice, args.modo, args.top, args.log, destino)


if __name__ == '__main__':
    main()
