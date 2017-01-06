function [Network_Structure] = Network_Structure_Definition(InputSize, OutputSize, HiddenLayerNum)
%   This function is used to define the structure of multi-label neural
% network according to the size of input, output, and the number of
% hidden layers designated by user.
%   The number of nodes in the hidden layer is calculated out through
% empirical equation: int((InputSize + OutputSize) / 2), which is
% allowed to change in this .m file.
%   Generally, the network is considered to be fully-connected between
% two layers. Besides, the value of HiddenLayerNum had better not to be too
% large due to the possible gradient explosion and vanishing. One can 
% resort to deep learning methods with such large scale and complex tasks.

% Author: Yixuan Xu | yixuan_xu@outlook.com
% Latest Update Time: Jan 1, 2017

% Empirical Equation
HiddenLayerSize = floor((InputSize + OutputSize)/2);

% Bias units are included in the input layer and hidden layers
Network_Structure.InputSize = InputSize + 1; 
Network_Structure.HiddenSize = HiddenLayerSize + 1;
Network_Structure.HiddenLayerNum = HiddenLayerNum;
Network_Structure.OutputSize = OutputSize;

end

