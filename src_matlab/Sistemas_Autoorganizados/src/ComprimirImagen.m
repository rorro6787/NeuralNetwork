%Archivo='Baboon'%Archivo que se desea Comprimir
%Archivo='Peppers'%Archivo que se desea Comprimir
Archivo='Baboon' %Archivo que se desea Comprimir

fichero=[Archivo '.png'];

imagenRGB=imread(fichero);
subplot(2,2,1);imshow(uint8(imagenRGB));

[numFilas,numColum,Rgb]=size(imagenRGB);

muestras=reshape(imagenRGB,[numFilas*numColum 3]);
Muestras=double(muestras');

% El shape de Muestras es [3, 262144] porque la imagen es de 512x512 y tiene 3 canales de color

Model=EntrenarSOFM(Muestras,50,50,100000);
Ganadoras=CompeticionSOFM(Model,Muestras); % Esta hay que implementar
ganadoras=reshape(Ganadoras,[numFilas numColum]);
subplot(2,2,2);imshow(uint8(ganadoras));

B=Model.Medias(:,Ganadoras);
Comp=(reshape(B',[numFilas numColum 3]));
Comprimida=uint8(Comp);
subplot(2,2,4);imshow(Comprimida);
% imwrite(Comprimida,[Archivo 'SOFM.tif'],'tiff');


Mosaico=ConvertirModelImg(Model);
subplot(2,2,3);imshow(Mosaico);
% saveas(Mosaico,[Archivo 'SOFM.fig']) 


[RGB,MOD]=CalculaError(Model,Muestras,Ganadoras);
disp(['Error RGB: ' num2str(RGB)]);
disp(['Error MOD: ' num2str(MOD)]);