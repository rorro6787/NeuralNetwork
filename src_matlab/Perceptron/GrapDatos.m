function GrapDatos(Data,Limites)
Inp=Data(:,1:end-1);
Out=Data(:,end);
plot(Inp(Out==1,1),Inp(Out==1,2),'rx',"MarkerSize",12,"LineWidth",2);hold on;
plot(Inp(Out==-1,1),Inp(Out==-1,2),'go',"MarkerSize",12,"LineWidth",2);hold on;
axis(Limites)