clear all;

D = load('handwriting.mat');
X = D.X;

[N, K] = size(X);

% Etiqueta de clase
J = 10;

Y = zeros(N, J);

% Generate the Y Label
for i = 1:J
    Y(1 + (500 * (i - 1)):i * 500, i) = 1;
end

% Scale the data
Xscaled = (X - min(X)) ./ (max(X) - min(X));

% Remove the NaN elements
Xscaled = Xscaled(:, any(~isnan(Xscaled)));

% Compute again the number of total elements and attributes
[N, K] = size(Xscaled);

% Create the test set
CVHO = cvpartition(N, 'HoldOut', 0.25);

XscaledTrain = Xscaled(CVHO.training(1), :);
XscaledTest = Xscaled(CVHO.test(1), :);
YTrain = Y(CVHO.training(1), :);
YTest = Y(CVHO.test(1), :);

% Create the validation set
[NTrain, K] = size(XscaledTrain);
CVHOV = cvpartition(NTrain, 'HoldOut', 0.25);

% Generate the validation sets
XscaledTrainVal = XscaledTrain(CVHOV.training(1), :);
XscaledVal = XscaledTrain(CVHOV.test(1), :);
YTrainVal = YTrain(CVHOV.training(1), :);
YVal = YTrain(CVHOV.test(1), :);

% Performance Matrix
Performance = zeros(7, 6);

i = 0;
% Estimate the hyper-parameters values
for C = [10^(-3), 10^(-2), 10^(-1), 1, 10, 100, 1000]
    i = i + 1;
    j = 0; % Reset j for each value of C
    for L = [50, 100, 500, 1000, 1500, 2000]
        j = j + 1;
        
        % Parametros aleatorios del modelo neuronal en capa oculta
        % Trabajamos con XscaledTrainVal
        t = rand(L, K) * 2 - 1;
        u = t * XscaledTrainVal';
        H = 1 ./ (1 + exp(-u));
        I = eye(L);
        w = (I / C + H' * H) \ (H' * YTrainVal);

        % Trabajamos con XscaledVal
        Hval = 1 ./ (1 + exp(-t * XscaledVal'));
        YVale = Hval * w;

        % error <- YVal, YVale
        error = sum(sum(YVal == round(YVale))) / numel(YVal);

        % Implementar el ELM neuronal, calcular el rendimiento asociado a C
        % y L
        Performance(i, j) = error;
        
    end
end

C = [10^(-3), 10^(-2), 10^(-1), 1, 10, 100, 1000];
L = [50, 100, 500, 1000, 1500, 2000];

[maxValue, linearIndexesOfMaxes] = max(Performance(:));
[rowsOfMaxes, colsOfMaxes] = find(Performance == maxValue);

% Una vez estimados los valores de C y L, seleccionar el mejor 
% y entrenamos al modelo usando todos los datos de entrenamiento
Copt = C(rowsOfMaxes(1));
Lopt = L(colsOfMaxes(1));

% Calcular con el conjunto de entrenamiento el ELM neuronal y
% reportar el error cometido en test
t = rand(Lopt, K) * 2 - 1;
u = t * XscaledTrain';
H = 1 ./ (1 + exp(-u));
w = (I / Copt + H' * H) \ (H' * YTrain);
Htest = 1 ./ (1 + exp(-t * XscaledTest'));
YTeste = Htest * w;

% Calcula el accuracy entre YTest e YTeste:
errorTest = sum(sum(YTest == round(YTeste))) / numel(YTest);
