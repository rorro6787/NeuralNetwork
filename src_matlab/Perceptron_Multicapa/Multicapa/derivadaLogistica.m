function [Y] = derivadaLogistica(X, beta)
    %% Calcula la derivada de la función logística para cada uno de los elementos del vector columna X
    Y_log = logistica(X, beta);  % Compute the logistic function
    Y = Y_log .* (1 - Y_log);    % Calculate the derivative using the output of the logistic function
end