function Mosaico=ConvertirModelImg(Model)
NumFil=Model.NumFilasMapa;
NumCol=Model.NumColsMapa;
Mosaico=zeros(NumFil,NumCol,3);
for x=1:NumFil
    for y=1:NumCol
        Mosaico(x,y,1)=Model.Medias(1,x,y)./255;
        Mosaico(x,y,2)=Model.Medias(2,x,y)./255;
        Mosaico(x,y,3)=Model.Medias(3,x,y)./255;
    end
end