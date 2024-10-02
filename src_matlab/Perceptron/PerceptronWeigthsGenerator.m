function W=PerceptronWeigthsGenerator(Data)
NInp=size(Data,2);
W=rand(NInp,1)-0.5;