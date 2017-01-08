function [Weight_Matrix] = Weight_Matrix_Initialization(Network_Structure)
%   This function is used to initialize weight matrixes between layers
% according to the struct of network structure. All matrixes are stored in
% a struct (Weight_Matrix) after initialization.

% Link weights are initialized between (-eps, +eps)
eps = 1;

% Initialize link weight matrix between input layer and hidden layer
Weight_Matrix.InputToHidden = rand((Network_Structure.InputSize), (Network_Structure.HiddenSize-1)) * (2*eps) - eps; % Note that there is no link between input layer and the bias unit in hidden layer

% Initialize link weight matrix between hidden layers
if(Network_Structure.HiddenLayerNum > 1)
    % link weight matrix exists with more than one hidden layers
    for i = 1 : (Network_Structure.HiddenLayerNum-1) % A multi-label neural network with too many hidden layers is highly not recommended
        Weight_Matrix.HiddenToHidden(:,:,i) = rand(Network_Structure.HiddenSize, (Network_Structure.HiddenSize - 1)) * (2*eps) - eps; % Note that there is no link between the former hidden layer and the bias unit in the following hidden layer
    end
end

% Initialize link weight matrix between hidden layer and output layer
Weight_Matrix.HiddenToOutput = rand(Network_Structure.HiddenSize, Network_Structure.OutputSize) * (2*eps) - eps;
end

