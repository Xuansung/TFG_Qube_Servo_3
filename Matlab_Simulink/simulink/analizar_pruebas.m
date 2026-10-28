function tabla = analizar_pruebas(orden)
% Script para entender Qube Servo 3
% Como funciona con 500Hz, Hace el muestreo cada 2ms
% y la grabación sale completa hasta unos 9.99s.
%------------------------
% USAR 9.99s
% AMPLITUD 1.5 +-= 85.944º, es decir, no son 90º puros.
% POR DEFECTO ORDENA POR
% ITAE, AÑADIR ARGUMENTO P.E:('sobrepaso') para otro
%-------------------------------
% PRUEBA DE POSICIÓN.
 if nargin < 1 
    orden = 'ITAE';
end
orden = upper(orden);
% VARIABLES DE ESTUDIO
ScopeData = evalin('base', 'ScopeData');
ScopeData1 = evalin('base', 'ScopeData1');
Kp = evalin('base', 'Kp');
Kd = evalin('base', 'Kd');
Ts = 0.002;
t = ScopeData1(:,1); % tiempo en s
r = ScopeData1(:,2); % valor teórico(posición en rad)
y = ScopeData1(:,3); % valor real (posición en rad)
u = ScopeData(:,2); % voltaje que se manda al motor
% Para encontrar en qué momento se producen los cambios
cambios = find(abs(diff(r)) > 0.5); % Posiciones donde hay cambios
IAE = []; ITAE = []; SOBREPASO = [];TV = [];
for n = 2:numel(cambios)
    a = cambios(n-1) + 1; % n fila del tramo
    b = cambios(n);% n + 1 fila del tramo
    e = r(a:b) - y(a:b); % error solo de ese tramo
    tseg = t(a:b) - t(a); % tiempo del tramo
    IAE(n-1) = sum(abs(e)) * Ts;
    ITAE(n-1) = sum(tseg.*abs(e)) * Ts;
    % Cálculo del sobrepaso
    salto = r(a) - y(a);
    yn = (y(a:b) - y(a)) / salto;
    SOBREPASO(n-1) = max(0, max(yn) - 1) * 100;
    TV(n-1) = sum(abs(diff(u(a:b))));
end
fila = [Kp Kd mean(IAE) mean(ITAE) mean(SOBREPASO) mean(TV)];
% GUARDAR RESULTADOS
if isfile('pruebas/resultados.mat')
    load('pruebas/resultados.mat', 'resultados')
else
    resultados = [];
end
resultados = [resultados; fila];
save('pruebas/resultados.mat', 'resultados')
save(sprintf('pruebas/prueba_kp%g_kd%g.mat', Kp, Kd), 'ScopeData', 'ScopeData1', 'Kp', 'Kd')
% VER Y COMPARAR
tabla = array2table(resultados, 'VariableNames', {'Kp','Kd','IAE','ITAE','SOBREPASO','TV'});
if strcmp(orden, 'ITAE')
    tabla = sortrows(tabla, 'ITAE');
else
    tabla = sortrows(tabla, {orden, 'ITAE'});
end
end
