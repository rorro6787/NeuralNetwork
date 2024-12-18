% Número máximo de iteraciones
numIt = 21;

% Patrón a memorizar
s = [-1 -1 1 1 1];

% Inicialización de la matriz de pesos
w = (1/size(s, 2)) * s' * s;
w = w - diag(diag(w)); % Eliminar autoconexiones para que la red sea no lineal

% Inicialización del estado de la red en la primera iteración
S = zeros(size(s, 2), numIt);
S(:, 1) = [-1 -1 1 -1 1];

% Bucle de iteraciones
for t = 2:numIt
    cambio = false;
    S(:, t) = S(:, t - 1);
    for i = 1:size(s, 2)
        % Calcular la entrada neta para cada neurona
        h = sum(S(:, t)' .* w(i, :), 'all');
        % Aplicar función de activación
        S(i, t) = (h > 0) * 2 - 1;
        cambio = cambio || S(i, t) ~= S(i, t - 1);
    end
    % Comprobar si la red se ha estabilizado
    if ~cambio
        disp('Estado estabilizado:');
        disp(S(:, t))
        return
    end
end

% No hay ninguna entrada que haga que la red se estabilice en un estado diferente del patrón memorizado o de su opuesto. 
% La salida de la red es una función determinista del patrón de entrada y de los pesos.

% La red se estabiliza en su opuesto cuando la entrada neta es negativa para todas las neuronas. 
% Esto depende de la orientación de los pesos y del patrón a memorizar. En este caso, la red se estabilizará en -s.

