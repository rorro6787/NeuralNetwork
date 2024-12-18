function [MOD, RGB] = CalculaError (Model, Muestras, Ganadoras)
    R = norm(Muestras(1,:) - Model.Medias(1,Ganadoras),2);
    G = norm(Muestras(2,:) - Model.Medias(2,Ganadoras),2);
    B = norm(Muestras(3,:) - Model.Medias(3,Ganadoras),2);

    RGB = [R G B];
    MOD = sqrt(R^2 + G^2 + B^2);
end