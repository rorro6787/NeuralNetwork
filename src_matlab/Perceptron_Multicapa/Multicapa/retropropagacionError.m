function [difW, difT] = retropropagacionError(patron, Z, y, w, s, h, u, Beta, eta)
    %% Función que calcula los diferenciales de los pesos W y T
    
    %% Inicialización de variables
    nSalidas = size(y, 1);  % Número de neuronas de salida
    nOcultas = size(w, 2);  % Número de neuronas ocultas
    
    % Inicializar deltas y diferenciales
    delta2 = zeros(nSalidas, 1);
    difW = zeros(nSalidas, nOcultas);
    delta1 = zeros(nOcultas, 1);
    difT = zeros(nOcultas, size(patron, 2));

    %% Cálculo de deltas2 y difW
    % Error de la capa de salida
    delta2 = (y - Z') .* derivadaLogistica(u, Beta);  % Cálculo del delta para la capa de salida
    difW = delta2 * h';  % Gradient para actualizar los pesos w (de salida)

    %% Cálculo de deltas1 y difT
    % Calcular el delta para la capa oculta
    delta1 = (w' * delta2) .* derivadaLogistica(s, Beta);  % Cálculo del delta para la capa oculta
    difT = delta1 * patron;  % Gradient para actualizar los pesos t (de la capa oculta)
end