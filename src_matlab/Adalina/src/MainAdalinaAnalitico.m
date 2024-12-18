clear all;
clc;
close all;

load DatosAND
%load DatosLS5
%load DatosLS10
%load DatosLS50
%load DatosOR
%load DatosXOR

X = Data(:, 1:end-1);
Y = Data(:, end);
K = size(Data, 2)-1;
N = size(Data, 1);

Xext = [X - ones(N, 1)];
size(Xext)
Xest
W1 = inv(Xext' * Xext) * Y;
W2 = pinv(Xext) * Y;
W1
W2

ypred = Xext * W;
Label = Signo(ypred);
total = sum(Label == Y)/N;
total
