clear;
clc;
close all;

load DatosAND
%load DatosLS5
%load DatosLS10
%load DatosLS50
%load DatosOR
%load DatosXOR

LR=0.5;
Limites=[-1.5, 2.5, -1.5, 2.5];
MaxEpoc=30;

W=PerceptronWeigthsGenerator(Data);

Epoc=1;
ecm = zeros(MaxEpoc-1, 1);

while ~CheckPattern(Data,W) && Epoc<MaxEpoc
     et = 0;
     for i=1:size(Data,1)
        [Input,Output,Target]=ValoresIOT(Data,W,i);
        et = et + (Output - Target)^2;
        GrapDatos(Data,Limites);
        GrapPatron(Input,Output,Limites);
        GrapNeuron(W,Limites);hold off;
        drawnow
%         pause;

        if Signo(Output) ~= Target
            W=UpdateNet(W,LR,Output,Target,Input);
        end

        
        GrapDatos(Data,Limites);
        GrapPatron(Input,Output,Limites)
        GrapNeuron(W,Limites);hold off;
        drawnow
%         pause;
     
     end
     ecm(Epoc, 1) = et/size(Data, 1);
     Epoc=Epoc+1;
end

figure;
plot(1:Epoc-1, ecm(1:Epoc-1), 'LineWidth', 2);
xlabel('Epoch');
ylabel('ECM (Error Cuadrático Medio)');
title('Evolución del ECM a lo largo de las épocas');
grid on;


