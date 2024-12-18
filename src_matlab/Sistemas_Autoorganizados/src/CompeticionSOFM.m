% Muestras tiene shape [3, 262144]
function [Ganadoras]=CompeticionSOFM(Model, Muestras)
    NumMuestras = size(Muestras, 2);
    Ganadoras = zeros(1, NumMuestras); % Almacenamos la neurona ganadora para cada pixel

    % Calculamos el numero de filas y columnas de la rejilla de Kohonen y el numero total de neuronas
    NumFilasMapa = Model.NumFilasMapa;
    NumColsMapa = Model.NumColsMapa;
    NumTotalNeuronas = NumFilasMapa * NumColsMapa;

    for n=1:1:NumMuestras
        MiMuestra = Muestras(:, n); % MiMuestra son los valores RGB de un determinado pixel
        DistsCuadrado = sum((repmat(MiMuestra, 1, NumTotalNeuronas) - Model.Medias(:,:))).^2; % Calculamos la distancia cuadrada de MiMuestra a cada una de las neuronas
        [~,indexGanadora] = min(DistsCuadrado);
        Ganadoras(n) = indexGanadora;
    end
end