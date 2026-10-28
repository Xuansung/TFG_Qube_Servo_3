%Función para representar gráficamente los resultados
% Por defecto ordena por ITAE
% pero se puede añadir un
% argumento que es el nombre del índice que se quiere.
function grafica_indices(indice)
 if nargin < 1 
    indice = 'ITAE';
end
indice = upper(indice);
if ~ismember(indice, {'IAE', 'ITAE', 'SOBREPASO', 'TV'})
    error('Indice no valido: usa IAE, ITAE, SOBREPASO o TV.');
end
load('pruebas/resultados_medios.mat', 'medias')
columna = [indice '_media'];
figure
h = heatmap(medias, 'Kp', 'Kd', 'ColorVariable', columna);
if all(medias.(columna) > 0)
    h.ColorScaling = 'log';
end
h.CellLabelFormat = '%.3g';
h.YDisplayData = flipud(h.YDisplayData);
h.Colormap = interp1([0 .5 1], [0 .7 0; 1 .85 0; .85 0 0], linspace(0, 1, 256));
title([indice ' medio (verde = mejor, rojo = peor)'])
end
