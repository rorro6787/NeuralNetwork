function W = UpdateNet(W, LR, Output, Target, Input)
    diffW = LR * (Target-Output) * [Input -1]
    %W = W + LR * error * [Input-1]';
    % W tiene dimensión K+1 porque tiene el sesgo. Input tiene dimensión K
    % y lo hacemos tener dimensión K+1
    W = W + diffW';    
end

