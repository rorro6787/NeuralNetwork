clear all;clc;

d(:,:,1) = [1 1 -1 -1 -1 1 1;
             1 1 -1 1 -1 1 1;
             1 1 -1 1 -1 1 1;
             1 1 -1 1 -1 1 1;
             1 -1 -1 -1 -1 -1 1;
             1 -1 1 1 1 -1 1;
             1 -1 1 1 1 -1 1;
             1 -1 1 1 1 -1 1;
             -1 -1 1 1 1 -1 -1;];

d(:,:,2) = [1 -1 -1 -1 -1 -1 1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 -1 -1 -1 -1 1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 -1 -1 -1 -1 1;];

d(:,:,3) = [1 -1 -1 -1 -1 -1 -1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             1 -1 -1 -1 -1 -1 -1;];

d(:,:,4) = [1 -1 -1 -1 -1 -1 1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 1 1 1 1 -1;
             1 -1 -1 -1 -1 -1 1;];

d(:,:,5) = [-1 -1 -1 -1 -1 -1 -1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 -1 -1 -1 -1 -1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 1 1 1 1 1 1;
             -1 -1 -1 -1 -1 -1 -1;];

w = zeros(9*7, 9*7);
dVect = zeros(5, 9*7);

for i = 1:5
    dVect(i, :) = reshape(d(:,:,i), 1, 9*7);
    w = w + dVect(i, :)' * dVect(i, :);
end

w = (1/size(w,1)) * w;
w = w - diag(diag(w));

numIt = 21;

for k = 1:5
    S = zeros(size(w,1), numIt);
    t = 1;
    S(:, t) = dVect(k, :);
    disp("Modelo inicial para el patrón " + char('A' + k - 1))
    disp(reshape(S(:, t), 9, 7))
    
    for t = 2:numIt
        cambio = false;
        S(:, t) = S(:, t-1);
        
        for i = 1:size(S, 2)
            h = sum(S(:, t)' .* w(i, :), 'all');
            S(i, t) = (h > 0) * 2 - 1;
            cambio = cambio || S(i, t) ~= S(i, t-1);
        end
        
        if ~cambio
            disp("Modelo final para el patrón " + char('A' + k - 1))
            disp(reshape(S(:, t), 9, 7))
            return
        end
    end
end