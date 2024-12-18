clear all;

X = rand(4, 5);
Y = rand(4, 2);

A = rand(4, 5);
B = rand(4, 5);

X = orth(X')';
Y = orth(Y);

w1 = X' * Y;
w2 = A' * B;
