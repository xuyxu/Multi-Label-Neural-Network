function [Weight_Matrix] = Weight_Matrix_Update(alpha, dj, es, Network_Structure, Weight_Matrix, Neural_Network_IOs, Input)
%    This function is used to update link weights in the multi-label neural
%  network after back propagating.

% Make sure the value of alpha is between 0 and 1
if(Check(alpha,0,1) || Check(alpha,1,2))
    error('The values of learning rate alpha is out of range!');
end

% Update link weights between the last hidden layer and the output layer
Weight_Matrix.HiddenToOutput = Weight_Matrix.HiddenToOutput + alpha * [1;Neural_Network_IOs{2,end-1}] * dj';

% Update link weights between hidden layers if exists
if(Network_Structure.HiddenLayerNum > 1)
    for i = 1 : (Network_Structure.HiddenLayerNum-1)
        Weight_Matrix.HiddenToHidden(:,:,(Network_Structure.HiddenLayerNum-i)) = Weight_Matrix.HiddenToHidden(:,:,(Network_Structure.HiddenLayerNum-i)) + alpha * [1;Neural_Network_IOs{2,Network_Structure.HiddenLayerNum-i}] * es(2:end,i)';
    end
end

% Update link weights between the input layer and the first hidden layer
Weight_Matrix.InputToHidden  = Weight_Matrix.InputToHidden + alpha * [1;Input] * es(2:end,1)';
end

