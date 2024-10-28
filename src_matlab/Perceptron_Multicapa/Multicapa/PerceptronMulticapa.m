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
%  load('Regresion/D_Parkinson_Telemonit.mat');Neu=[21 1];
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
load('Clasificacion/D_Vertebral_Column.mat');Neu=[6 1];

%% Configuración de la red
Beta=1;
eta=0.1;
maxEpoch=3000;

%% Inicialización de variables

%%Lectura de datos
X=data(:,1:end-1);
Z=data(:,end);

%Arquitectura de la red
nOcultas=Neu(1);%Numero de neuronas ocultas
nSalidas=size(Z,2);%Numero de neuronas salidas

%Pesos de la capa oculta y de salida
t=rand(nOcultas,size(X,2)); %número de ocultas x tantos pesos como entradas X 
w=rand(nSalidas,nOcultas); %número de salidas x tantos pesos como salidas de la capa oculta X 

%% Distribución aleatoria de los datos en Entrenamiento, Validación y Test
numEntrenamiento=round(size(X,1)*0.5);
numValidacion=round(size(X,1)*0.2);
numTest=round(size(X,1)*0.3);
idxPermutacion=randperm(size(X,1));
idxEntrenamiento=idxPermutacion(:,1:numEntrenamiento);
idxValidacion=idxPermutacion(:,numEntrenamiento+1:numEntrenamiento+numValidacion);
idxTest=idxPermutacion(:,numEntrenamiento+numValidacion+1:end);
%%

MSEAcumuladoValid_vector=zeros(maxEpoch, 1);
MSEAcumuladoEntrena_vector=zeros(maxEpoch, 1);
MSEAcumuladoTest_vector=zeros(maxEpoch, 1);
for epoch=1:maxEpoch
    %eta=((maxEpoch-epoch)/maxEpoch)*0.05+0.1; %Tasa de aprendizaje
    %decreciente
    
    %% << ETAPA DE ENTRENAMIENTO >>
    for idx=idxEntrenamiento
        %% Cálculo de salida de la red para entrenamiento
        [y, h, s, u]=salidaRed(X(idx,:), t, w, Beta);
        MSEAcumuladoEntrena_vector(epoch)=MSEAcumuladoEntrena_vector(epoch)+sum((Z(idx,:)'-y(:)).^2, "all");
        
        %% Backpropagation
        [difW, difT] = retropropagacionError(X(idx,:), Z(idx,:), y, w, s, h, u, Beta, eta);
        
        %% Actualizacion de pesos
        w=w+difW;
        t=t+difT;
    end
    %% Cálculo de MSE en entrenamiento
    MSEAcumuladoEntrena_vector(epoch)=MSEAcumuladoEntrena_vector(epoch)/size(idxEntrenamiento,2);   
    
    %% << ETAPA DE VALIDACIÓN >>
    for idx=idxValidacion
        %% Cálculo de salida de la red para validación
        [y, h, s, u]=salidaRed(X(idx,:), t, w, Beta);
        MSEAcumuladoValid_vector(epoch)=MSEAcumuladoValid_vector(epoch)+sum((Z(idx,:)'-y(:)).^2, "all");
    end
    MSEAcumuladoValid_vector(epoch)=MSEAcumuladoValid_vector(epoch)/size(idxValidacion,2);
    
    %% << ETAPA DE TEST >>
    %%Nota: realmente no sería necesario calcular el MSE para cada época, sólo sería necesario cuando 
    %%terminemos de entrenar la red, es decir, ...
    for idx=idxTest
        %% Cálculo de salida de la red para test
        [y, h, s, u]=salidaRed(X(idx,:), t, w, Beta);
        MSEAcumuladoTest_vector(epoch)=MSEAcumuladoTest_vector(epoch)+sum((Z(idx,:)'-y(:)).^2, "all");
    end
    MSEAcumuladoTest_vector(epoch)=MSEAcumuladoTest_vector(epoch)/size(idxTest,2);
end

plot(MSEAcumuladoEntrena_vector,'b')
xlabel('Época')
hold on
plot(MSEAcumuladoValid_vector,'g')
ylabel('ECM')
hold on
%% --> Pintar el MSE del conjunto de test <--
%% ->> Completar aquí <<-

