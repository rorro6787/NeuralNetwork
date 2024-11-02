function [y, h, s, u] = salidaRed(patron, t, w, Beta)
    %% Función que calcula la salida de la red (y), los pesos (h, s) y la salida de la capa oculta (s)
    
    % Calculo de las salidas de la capa oculta
    s = t * patron';      % Producto de pesos de la capa oculta y entrada
    h = logistica(s, Beta); % Aplicar función logística a las entradas de la capa oculta

    % Calculo de la salida de la red
    u = w * h;            % Producto de pesos de la capa de salida y salida de la capa oculta
    y = logistica(u, Beta); % Aplicar función logística a la salida de la red
end