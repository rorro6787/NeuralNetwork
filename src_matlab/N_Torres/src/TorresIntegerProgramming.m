clear all;

N = 8;
f = -ones(N*N,1);

A = [];

for i = 1:1:N
    W = zeros(N,N);
    W(i,1:N) = 1;
    A = [A; reshape(W,1,[])];
    W = zeros(N,N);
    W(1:N,i) = 1;
    A = [A; reshape(W,1,[])];
end

size(A)
b = ones(N+N,1);
intcon = 1:1:N*N;

x = intlinprog(f,intcon,[],[],A,b,zeros(N*N,1),ones(N*N,1));

result = reshape(x,N,N)