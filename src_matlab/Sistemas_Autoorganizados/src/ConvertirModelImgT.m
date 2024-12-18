function Mosaico=ConvertirModelImgT(Model,Tam)
NumFil=Model.NumFilasMapa;
NumCol=Model.NumColsMapa;
Mosaico=zeros(Tam(1),Tam(2),3);

for NdxFila=1:NumFil
    RangoFilas=(Tam(1)./NumFil*(NdxFila-1)+1):Tam(1)./NumFil*NdxFila;
    for NdxCol=1:NumCol
        RangoCols=(Tam(2)./NumCol*(NdxCol-1)+1):Tam(2)./NumCol*NdxCol; 
        Mosaico(RangoFilas,RangoCols,1)=Model.Medias(1,NdxFila,NdxCol)*ones(Tam(1)./NumFil,Tam(2)./NumCol)/255;
        Mosaico(RangoFilas,RangoCols,2)=Model.Medias(2,NdxFila,NdxCol)*ones(Tam(1)./NumFil,Tam(2)./NumCol)/255;
        Mosaico(RangoFilas,RangoCols,3)=Model.Medias(3,NdxFila,NdxCol)*ones(Tam(1)./NumFil,Tam(2)./NumCol)/255;
    end
end