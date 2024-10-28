function result = CheckPattern(Data, W)
    % Inicializa una variable para comprobar si todos los patrones están correctos
    correctly_classified = true;
    for i = 1:1:size(Data, 1)
        % Obtiene los valores de entrada, salida y objetivo
        [Input, Output, Target] = ValoresIOT(Data, W, i);
        
        % Si algún patrón no está correctamente clasificado, establece la variable a false
        if Output ~= Target
            correctly_classified = false;
            break;
        end
    end
    % Retorna el resultado de la comprobación
    result = correctly_classified;
end

