clear all;

D = load('handwriting.mat');
X = D.X;

[N, K] = size(X);
J = 10;

Y = zeros(N,J);

% Generar etiquetas Y
for i =1:10
    Y(1+(500*(i-1)):i*500,i) =1;
end

% Escalar los datos
Xscaled = (X-min(X))./(max(X)-min(X));
Xscaled = Xscaled(:,any(~isnan(Xscaled)));

[N, K] = size(Xscaled);

CVHO = cvpartition(N,'HoldOut',0.25);

XscaledTrain = Xscaled(CVHO.training(1),:);
XscaledTest = Xscaled(CVHO.test(1),:);
YTrain = Y(CVHO.training(1),:);
YTest = Y(CVHO.test(1),:);

% Crear conjunto de validación
[NTrain, K] = size(XscaledTrain);
CVHOV = cvpartition(NTrain,'HoldOut',0.25);

XscaledTrainVal = XscaledTrain(CVHOV.training(1),:);
XscaledVal = XscaledTrain(CVHOV.test(1),:);
YTrainVal = YTrain(CVHOV.training(1),:);
YVal = YTrain(CVHOV.test(1),:);

% Matriz de rendimiento
Performance = zeros(7,6);

% Parámetro del kernel RBF
gamma = 0.1;

% Definir función RBF para el kernel
rbf_kernel = @(X, Y) exp(-gamma * (pdist2(X, Y, 'euclidean').^2));

i = 0;
j = 0;

% Estimación de hiperparámetros
for C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000]
    i = i+1;
    for L = [50 100 500 1000 1500 2000]
        j = j+1;

        % Generar pesos de la capa oculta y aplicar kernel
        W_hidden = randn(K, L);
        H = rbf_kernel(XscaledTrainVal, W_hidden');

        % Calcular pesos de salida con regularización
        W_output = (H' * H + (1/C) * eye(L)) \ (H' * YTrainVal);

        % Aplicar kernel RBF en validación
        H_val = rbf_kernel(XscaledVal, W_hidden');
        Y_pred_val = H_val * W_output;

        % CCR en conjunto de validación
        CCR_val = sum(vec2ind(Y_pred_val') == vec2ind(YVal')) / size(YVal, 1);
        
        % Guardar rendimiento (CCR)
        Performance(i, j) = CCR_val;
        
    end
    j = 0;
end

C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000];
L = [50 100 500 1000 1500 2000];

[maxValue, linearIndexesOfMaxes] = max(Performance(:));
[rowsOfMaxes, colsOfMaxes] = find(Performance == maxValue);

Copt = C(rowsOfMaxes(1));
Lopt = L(colsOfMaxes(1));

% Entrenamiento final con C y L óptimos
W_hidden_opt = randn(K, Lopt);
H_opt = rbf_kernel(XscaledTrain, W_hidden_opt');
W_output_opt = (H_opt' * H_opt + (1/Copt) * eye(Lopt)) \ (H_opt' * YTrain);

% Predicciones en el conjunto de prueba
H_test = rbf_kernel(XscaledTest, W_hidden_opt');
Y_pred_test = H_test * W_output_opt;

% CCR en prueba
CCR_test = sum(vec2ind(Y_pred_test') == vec2ind(YTest')) / size(YTest, 1);

% MSE en prueba
MSE_test = mean((Y_pred_test - YTest).^2, 'all');

% Mostrar resultados
fprintf('Mejor C: %f\n', Copt);
fprintf('Mejor L: %d\n', Lopt);
fprintf('CCR en Test: %f\n', CCR_test);
fprintf('MSE en Test: %f\n', MSE_test);