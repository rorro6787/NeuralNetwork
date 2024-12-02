clear all;

load('barco.mat');
load('coche.mat');
load('textoBarco.mat');
load('textoCoche.mat');

X(1, :) = reshape(barco, 1, size(barco, 1)*size(barco, 2));
X(2, :) = reshape(coche, 1, size(coche, 1)*size(coche, 2));
Y(1, :) = reshape(textoBarco, 1, size(textoBarco, 1)*size(textoBarco, 2));
Y(2, :) = reshape(textoCoche, 1, size(textoCoche, 1)*size(textoCoche, 2));

% Regla de Hebb
W = X'*Y;  

% size(X, 2) = k
s = zeros(size(X, 2), 21);

% size(Y, 2) = j
s2 = zeros(size(Y, 2), 21);

s_init = reshape(barco, 1, size(barco, 1)*size(barco, 2));
s_init2 = sign(s_init*W);

s(:, 1) = s_init;
s2(:, 1) = s_init2;

for epoch= 2:1:21
    s(:, epoch) = sign(W*s2(:, epoch-1));
    s2(:, epoch) = sign(s(:, epoch-1)'*W);
    if sum(s(:, epoch) == s(:, epoch-1)) == size(X, 2) && sum(s2(:, epoch) == s2(:, epoch-1)) == size(Y, 2)
        subplot(3, 1, 1)
        imshow(reshape(s_init, size(barco, 1), size(barco, 2)))
        subplot(3, 1, 2)
        imshow(reshape(s2(:, epoch), size(textoBarco, 1), size(textoBarco, 2)))
        subplot(3, 1, 3)
        imshow(reshape(s(:, epoch), size(barco, 1), size(barco, 2)))
        return
    end
end



