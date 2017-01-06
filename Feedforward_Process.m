function [Output, Neural_Network_IOs] = Feedforward_Process(Input,Network_Structure, Weight_Matrix)
%   This function is used to realize feedforward process of multi-label
% neural network through matrix product, completed by CPU. Codes on GPU
% side are included in future plans due to the current poor compatibility
% of MATLAB (R2016a) and PASCAL series GPUs (with CUDA 7.5 or CUDA 8.0).

% Designate activation method
activation_function = 'tanh';

% Feedforward process
for i = 1 : (Network_Structure.HiddenLayerNum+1) % the number of link weight matrixes equals to (HiddenLayerNum+1)
    
    if (i == 1) % Store the input and output of the first hidden layer
        
        Neural_Network_IOs{1,i} = ([1;Input]' * Weight_Matrix.InputToHidden)'; % Size: (HiddenSize - 1) * 1
        Neural_Network_IOs{2,i} = Activation_Function(Neural_Network_IOs{1,i}, activation_function); % Size: (HiddenSize - 1) * 1
        
    elseif((i > 1) && (i < (Network_Structure.HiddenLayerNum + 1))) % Store the input and output of hidden layers except for the first hidden layer
        
        Neural_Network_IOs{1,i} = ([1;Neural_Network_IOs{2,i-1}]' * squeeze(Weight_Matrix.HiddenToHidden(:,:,(i-1))))'; % Size: (HiddenSize - 1) * 1
        Neural_Network_IOs{2,i} = Activation_Function(Neural_Network_IOs{1,i}, activation_function); % Size: (HiddenSize - 1) * 1
        
    else % Store the input and output of output layer
        
        Neural_Network_IOs{1,i} = ([1;Neural_Network_IOs{2,i-1}]' * Weight_Matrix.HiddenToOutput)'; % Size: OutputSize * 1
        Neural_Network_IOs{2,i} = Activation_Function(Neural_Network_IOs{1,i}, activation_function); % Size: OutputSize * 1
        
    end
end

Output = Neural_Network_IOs{2,(Network_Structure.HiddenLayerNum+1)};

end

