function GrapPatron(Inp,Out,Limites)
if Out==1
    plot(Inp(1),Inp(2),'bx',"MarkerSize",12,"LineWidth",2);hold on;
else
    plot(Inp(1),Inp(2),'bo',"MarkerSize",12,"LineWidth",2);hold on;
end
axis(Limites)