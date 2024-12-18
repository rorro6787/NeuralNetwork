clear all;
N = 3;
w = zeros(N,N,N,N);     % Las 2 primeras N te dan la posicion del tablero, y las otras 2 N te da como esta conectada con el resto del tablero
theta = zeros(N,N);     % Para cada casilla del tablero, el umbral que tienes para dicha neurona
theta(:,:) = -1;

for i = 1:N
    for j = 1:N
        w(i,j,i,1:N) = -2;  % Para cada neurona de la fila i, la conecto con todas las neuronas de la fila i
        w(i,j,1:N,j) = -2;  % Para cada neurona de la columna j, la conecto con todas las neuronas de la columna j
        w(i,j,i,j) = 0;     % No se conecta consigo misma
    end
end

% Dinamica de computacion
nEpochs = 20;
sList = zeros(N,N,nEpochs);

% Inicializacion del tablero
sList(:,:,1) = [0 1 1; 1 1 0; 0 1 0];
for e = 2:1:nEpochs
    disp(['Epoch: ', num2str(e)]);
    cambio = false;
    sList(:,:,e) = sList(:,:,e-1);
    for i = 1:N
        for j = 1:N
            h = 0;
            for k = 1:N
                for l = 1:N
                    h = h + w(i,j,k,l)*sList(k,l,e);
                end
            end
            sList(i,j,e) = int16(h >= theta(i,j));
            cambio = cambio || (sList(i,j,e) ~= sList(i,j,e-1));
        end
    end

    if ~cambio
        sList(:,:,e)
        return;
    end

end