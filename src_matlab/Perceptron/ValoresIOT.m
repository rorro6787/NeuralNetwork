function [Input, Output, Target] = ValoresIOT(Data, W, i)
    % Entrada del patrón actual (incluyendo el sesgo)
    Input = Data(i, 1:end-1);
    
    % Objetivo del patrón actual
    Target = Data(i, end);
    
    % Cálculo de la salida de la neurona (producto punto)
    Output = Signo(Input * W(1:end-1) - W(end));  % Función escalón
    %                X   @     W      -  sesgo
end

