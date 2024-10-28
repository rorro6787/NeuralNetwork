


clc;
clear;
File='Vertebral_Column';
archivo=['T_' File '.txt'];
data=load(archivo);
MaxData=max(data);
MinData=min(data);
for i=1:size(data,2)
    data(:,i)=data(:,i)-MinData(i);
    if MaxData(i)~=MinData(i)
        data(:,i)=data(:,i)./(MaxData(i)-MinData(i));
    end
end

save (['D_' File '.mat'], 'data')