% Introduction:
%   Each sample in the dataset can be represented by a feature vector with
% dimension 121*1. The total number of labels is 12. Besides, the neural
% newwork has two hidden layers.

% Load dataset
load 'DataSet.mat';

% One variable used to record loss after each iteration
Loss_Record = zeros(1,size(x,2));

% Designate the structure of multi-label nerual network
[Network_Structure] = Network_Structure_Definition(121, 12, 2);

% Designate learning rate
alpha = 0.5;

% Initialize all weight matrixes needed.
[Weight_Matrix] = Weight_Matrix_Initialization(Network_Structure);

% Start Trainging Process
for num = 1 : size(x,2)
    % Feedforward process
    [Output,Neural_Network_IOs] = Feedforward_Process(x(:,num), Network_Structure, Weight_Matrix);
    
    % Record the loss
    Loss_Record(1,num) = Loss_Function(Output, y(:,num));
    
    % Back propagation process
    % dj: error of output layer
    % es: error of hidden layers
    [dj, es] = BackPropagation_Process(Network_Structure, Weight_Matrix, Neural_Network_IOs, Output, y(:,num));
    
    % Update link weights according to error
    [Weight_Matrix] = Weight_Matrix_Update(alpha, dj, es, Network_Structure, Weight_Matrix, Neural_Network_IOs, x(:,num));
end

plot((1:size(x,2)),Loss_Record);xlabel('Iteration Time');ylabel('Loss');