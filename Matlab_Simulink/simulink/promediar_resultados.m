function promediar_resultados(orden)
 if nargin < 1 
    orden = 'ITAE';
end
orden = upper(orden);
load('pruebas/resultados.mat', 'resultados')
nombres = {'IAE', 'ITAE', 'SOBREPASO', 'TV'};
[combos, ~, g] = unique(resultados(:, 1:2), 'rows');
n = accumarray(g, 1);
medias = table(combos(:, 1), combos(:, 2), n, 'VariableNames', {'Kp', 'Kd', 'pruebas'});
for k = 1:4
    dev = accumarray(g, resultados(:, k+2), [], @std);
    dev(n == 1) = NaN;
    medias.([nombres{k} '_media']) = accumarray(g, resultados(:, k+2), [], @mean);
    medias.([nombres{k} '_std']) = dev;
end
if strcmp(orden, 'ITAE')
    medias = sortrows(medias, 'ITAE_media')
else
    medias = sortrows(medias, {[orden '_media'], 'ITAE_media'})
end
save('pruebas/resultados_medios.mat', 'medias')
writetable(medias, 'pruebas/resultados_medios.csv')
end
