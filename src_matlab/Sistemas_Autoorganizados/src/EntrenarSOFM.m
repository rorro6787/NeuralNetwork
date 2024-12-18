function [Modelo]=EntrenarSOFM(Muestras,NumFilasMapa,NumColsMapa,NumEtapas)
% Entrenar un modelo SOFM de Kohonen:
[Dimension,NumMuestras]=size(Muestras);

% Inicializacion
fprintf('Inicializando SOFM')
NumNeuro=NumFilasMapa*NumColsMapa;
Modelo.NumColsMapa=NumColsMapa;
Modelo.NumFilasMapa=NumFilasMapa;
Modelo.Dimension=Dimension;
NumPatIni=max([Dimension+1,ceil(NumMuestras/(NumFilasMapa*NumColsMapa))]);
Modelo.Medias=zeros(Dimension,NumFilasMapa,NumColsMapa);

for NdxFila=1:NumFilasMapa
    for NdxCol=1:NumColsMapa
        MisMuestras=Muestras(:,ceil(NumMuestras*rand(1,NumPatIni)));
        Modelo.Medias(:,NdxFila,NdxCol)=mean(MisMuestras')';  
        fprintf('.')
    end
end

[TodasXCoords TodasYCoords]=ind2sub([NumFilasMapa NumColsMapa],1:NumNeuro);
TodasCoords(1,:)=TodasXCoords;
TodasCoords(2,:)=TodasYCoords;
for NdxNeuro=1:NumNeuro    
    DistTopol{NdxNeuro}=sum((repmat(TodasCoords(:,NdxNeuro),1,NumNeuro)-TodasCoords).^2,1);
end

%Entrenamiento
fprintf('\nEntrenando SOFM\n')
MaxRadio=(NumFilasMapa+NumColsMapa)/8;
for NdxEtapa=1:NumEtapas
    MiMuestra=Muestras(:,ceil(NumMuestras*rand(1)));
    if NdxEtapa<0.5*NumEtapas   
        % Fase de ordenación: caída lineal
        TasaAprendizaje=0.4*(1-NdxEtapa/NumEtapas);
        MiRadio=MaxRadio*(1-(NdxEtapa-1)/NumEtapas);
    else
        % Fase de convergencia: constante
        TasaAprendizaje=0.01;
        MiRadio=1;
    end
    
    DistsCuadrado=sum((repmat(MiMuestra,1,NumNeuro)-Modelo.Medias(:,:)).^2,1);
    [Minimo NdxGana]=min(DistsCuadrado);
    Coef=repmat(TasaAprendizaje*exp(-DistTopol{NdxGana}/(MiRadio^2)),Dimension,1);
    % Actualizar las neuronas
    Modelo.Medias(:,:)=Coef.*repmat(MiMuestra,1,NumNeuro)+...
        (1-Coef).*Modelo.Medias(:,:);
    if mod(NdxEtapa,10000)==0
        fprintf('%d etapas completadas\n',NdxEtapa);
    end
end

fprintf('Entrenamiento finalizado\n')

    
    
        
