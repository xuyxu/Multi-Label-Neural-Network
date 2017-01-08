function [dj, es] = BackPropagation_Process(Network_Structure, Weight_Matrix, Neural_Network_IOs, Output, Target_output)
%   This function runs an adapted back propagation algorithm in multi-label
% neural network. More information is available in the original paper.
% Basically, the error at the output layer is back propagated to the
% hidden layers iteratively through chain rule.

% Calculate error at the output layer
dj = zeros(Network_Structure.OutputSize, 1); % pre-allocate memory

%   Due to the error function of the output layer is segmented, a loop is
% used to calculate error of each output node.
for i = 1 : Network_Structure.OutputSize
    
    % 1st situation: ith label is related with this sample
    if(Target_output(i,1) == 1) 
        temp_index = find(Target_output == -1); % find set l, which indicates all labels not belong to this sample
        temp_sum = 0;
        for j = 1 : size(temp_index,1)
            temp_sum = temp_sum + exp(-(Output(i,1) - Output(temp_index(j,1),1)));
        end
        temp_sum = temp_sum / (size(temp_index,1)) / (size(Output,1)-size(temp_index,1)); % normalize it by cardinality |yi| and |~yi|
        dj(i,1) = temp_sum * (1+Neural_Network_IOs{2,end}(i,1)) * (1-Neural_Network_IOs{2,end}(i,1));
        
    % 2nd situation: ith label is not related with this sample
    else 
        temp_index = find(Target_output == 1); % find set k, which indicates all labels belong to this sample
        temp_sum = 0;
        for j = 1 : size(temp_index,1)
            temp_sum = temp_sum + exp(-(Output(temp_index(j,1),1) - Output(i,1)));
        end
        temp_sum = (-temp_sum) / (size(temp_index,1)) / (size(Output,1)-size(temp_index,1));
        dj(i,1) = temp_sum * (1+Neural_Network_IOs{2,end}(i,1)) * (1-Neural_Network_IOs{2,end}(i,1));
    end
end
        
% Calculate error between hidden layers if exists
%   Since the error function is not segmented, error of hidden layers and
% input layer can be efficiently calculated through matrix product. 
if(Network_Structure.HiddenLayerNum > 1)
    for i = 1 : (Network_Structure.HiddenLayerNum)
        if(i == 1) % refers to the hidden layer just ahead of the output layer
            es(:,i) = (Weight_Matrix.HiddenToOutput * dj) .* (1+[1;Neural_Network_IOs{2,Network_Structure.HiddenLayerNum}]) .* (1-[1;Neural_Network_IOs{2,Network_Structure.HiddenLayerNum}]); % Size: HiddenSize * 1
        else
            % the first row of es indicates error of bias units, which is not needed when back propagating
            es(:,i) = (squeeze(Weight_Matrix.HiddenToHidden(:,:,(Network_Structure.HiddenLayerNum+1-i))) * es(2:end,i-1)) .* (1+[1;Neural_Network_IOs{2,Network_Structure.HiddenLayerNum+1-i}]) .* (1-[1;Neural_Network_IOs{2,Network_Structure.HiddenLayerNum+1-i}]);
        end
    end
end
% One thing worth mentioning is that the column order is inverse. 

