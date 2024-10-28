close all
clear all

%% Elección de conjunto de entrada
% load('Regresion/D_Airfoil.mat');Neu=[5  1]; 
% load('Regresion/D_C_C_Power_Plant.mat');Neu=[4  1]; 
% load('Regresion/D_Concrete_Com_Str.mat');Neu=[8 1];
% load('Regresion/D_Concrete_Slump.mat');Neu=[10 1]; 
% load('Regresion/D_Energy_Efficient_A.mat');Neu=[8 1];
 %load('Regresion/D_Energy_Efficient_B.mat');Neu=[8 1];
% load('Regresion/D_Energy_Efficient_B.mat');Neu=[8 1];
% load('Regresion/D_Forest_Fire.mat');Neu=[12 1];
% load('Regresion/D_Housing.mat');Neu=[13 1];
  load('Regresion/D_Parkinson_Telemonit.mat');Neu=[21 1];
%  load('Regresion/D_WineQuality_Red.mat');Neu=[11 1];
%  load('Regresion/D_WineQuality_White.mat');Neu=[11 1];
% load('Regresion/D_Yacht_Hydrodynamics.mat');Neu=[6  1]; 

% load('Clasificacion/D_Blood_Transfusion.mat');Neu=[4  1]; 
% load('Clasificacion/D_Cancer.mat');Neu=[9  1]; 
% load('Clasificacion/D_Card.mat');Neu=[51 1];
% load('Clasificacion/D_Climate.mat');Neu=[18 1]; 
% load('Clasificacion/D_Diabetes.mat');Neu=[8 1];
% load('Clasificacion/D_heartc.mat');Neu=[35 1];
% load('Clasificacion/D_Ionosphere.mat');Neu=[34 1];
% load('Clasificacion/D_Sonar.mat');Neu=[60 1];
% load('Clasificacion/D_Statlog(Heart).mat');Neu=[13 1];
%load('Clasificacion/D_Vertebral_Column.mat');Neu=[6 1]; 

%% Configuración de la red
Beta=1;
eta=0.1;
maxEpoch=3000;

%% Inicialización de variables

%%Lectura de datos
X=data(:,1:end-1);
Z=data(:,end);
%Z=[Z ~Z];  %Hacemos esto para probar con dos salidas

%Arquitectura de la red
nOcultas=Neu(1);%Numero de neuronas ocultas
nSalidas=size(Z,2);%Numero de neuronas salidas

%Pesos de la capa oculta y de salida
t=rand(nOcultas,size(X,2)); %número de ocultas x tantos pesos como entradas X 
w=rand(nSalidas,nOcultas); %número de salidas x tantos pesos como salidas de la capa oculta X 

%% Distribución aleatoria de los datos en nPliegues pliegues: Entrenamiento(90%) y Validación(10%)
nPliegues=10;
cardPliegue=floor(size(X,1)/nPliegues);
idxValidacion=zeros(cardPliegue,nPliegues);
idxEntrenamiento=zeros(cardPliegue*(nPliegues-1),nPliegues);
idxPermutacion=randperm(size(X,1));
idxPermutacion=idxPermutacion(1:end-mod(size(X,1),nPliegues));%Eliminamos los últimos para que sea múltiplos exactos de nPliegues
for i=1:nPliegues
    idxValidacion(:,i)=idxPermutacion(:,(i-1)*cardPliegue+1:i*cardPliegue);
end

idxEntrenamiento(:,1)=idxPermutacion(:,cardPliegue+1:end);
idxEntrenamiento(:,end)=idxPermutacion(:,1:(nPliegues-1)*cardPliegue);

for i=2:nPliegues-1
    idxEntrenamiento(:,i)=[idxPermutacion(:,1:(i-1)*cardPliegue) idxPermutacion(:,i*cardPliegue+1:nPliegues*cardPliegue)]';
end
%%

MSEAcumuladoValid_vector=zeros(maxEpoch, nPliegues);
MSEAcumuladoEntrena_vector=zeros(maxEpoch, nPliegues);
for idxPliegue=1:nPliegues
    fprintf('Pliegue: %d\n', idxPliegue);
    
    %% Inicializamos red en cada pliegue
    t=rand(nOcultas,size(X,2)); %número de ocultas x tantos pesos como entradas X
    w=rand(nSalidas,nOcultas); %número de salidas x tantos pesos como salidas de la capa oculta X
    %%
    
    for epoch=1:maxEpoch      
        MSEAcumuladoEntrena=0;
        
        %% << ETAPA DE ENTRENAMIENTO >>
        for idx=idxEntrenamiento(:,idxPliegue)'
            %% Cálculo de salida de la red para entrenamiento
            [y, h, s, u]=salidaRed(X(idx,:), t, w, Beta);
            MSEAcumuladoEntrena=MSEAcumuladoEntrena+sum((Z(idx,:)'-y(:)).^2, "all");
            
            %% Backpropagation
            [difW, difT] = retropropagacionError(X(idx,:), Z(idx,:), y, w, s, h, u, Beta, eta);
            
            %% Actualizacion de pesos
            w=w+difW;
            t=t+difT;
        end
        %% Cálculo de MSE en entrenamiento
        MSEAcumuladoEntrena=MSEAcumuladoEntrena/size(idxEntrenamiento,1);
        MSEAcumuladoEntrena_vector(epoch,idxPliegue)=MSEAcumuladoEntrena;
        
        %% << ETAPA DE VALIDACIÓN >>
        MSEAcumuladoValid=0;
        for idx=idxValidacion(:,idxPliegue)'
            %% Cálculo de salida de la red para validación
            [y, h, s, u]=salidaRed(X(idx,:), t, w, Beta);
            MSEAcumuladoValid=MSEAcumuladoValid+sum((Z(idx,:)'-y(:)).^2, "all");
        end
        MSEAcumuladoValid=MSEAcumuladoValid/size(idxValidacion,1);
        MSEAcumuladoValid_vector(epoch,idxPliegue)=MSEAcumuladoValid;
    end
end

plot(MSEAcumuladoEntrena_vector,'b')
hold on
plot(MSEAcumuladoValid_vector,'g')

MSE_medioPligues=mean(MSEAcumuladoValid_vector, 2);
hold on
plot(MSE_medioPligues,'r')


