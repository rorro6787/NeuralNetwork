clear;
clc;
% Data=[0 0 0;0 1 0; 1 0 0;1 1 1];%And
% save 'DatosAND.mat' 'Data'

% Data=[0 0 0;0 1 1; 1 0 1;1 1 1];
% save 'DatosOR.mat' 'Data'

% Data=[0 0 0;0 1 1; 1 0 1;1 1 0];
% save 'DatosXOR.mat' 'Data'

Npat=5;
X=rand(Npat,1);
Y=X-0.2;
Data(:,1)=X;
Data(:,2)=rand(Npat,1);
Data(:,3)=Y<Data(:,2);
save (['DatosLS' num2str(Npat) '.mat'], 'Data')
