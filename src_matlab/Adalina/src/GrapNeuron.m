function GrapNeuron(W,Limites)
X=Limites(1):0.1:Limites(2);
Y=(-W(1)*X+W(3))/W(2);
plot(X,Y,"MarkerSize",12,"LineWidth",2);hold on;
axis(Limites)