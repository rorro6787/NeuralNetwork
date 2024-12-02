clear all;

X(1, :) = [1 1 1 -1 1 -1 -1 1 -1];
X(2, :) = [1 -1 -1 1 -1 -1 1 1 1];
Y(1, :) = [1 -1 -1];
Y(2, :) = [1 -1 1];

% Regla de Hebb
W = X'*Y;  

% size(X, 2) = k
s = zeros(size(X, 2), 21);

% size(Y, 2) = j
s2 = zeros(size(Y, 2), 21);

s_init = [1 1 1 -1 1 -1 -1 1 -1];
s_init2 = sign(s_init*W);

s(:, 1) = s_init;
s2(:, 1) = s_init2;

for epoch= 2:1:21
    s(:, epoch) = sign(W*s2(:, epoch-1));
    s2(:, epoch) = sign(s(:, epoch-1)'*W);
    if sum(s(:, epoch) == s(:, epoch-1)) == size(X, 2) && sum(s2(:, epoch) == s2(:, epoch-1)) == size(Y, 2)
        s = s(:, epoch)
        s2 = s2(:, epoch)
        epoch
        break;
    end
end



