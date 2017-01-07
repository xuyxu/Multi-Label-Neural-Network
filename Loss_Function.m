function [loss] = Loss_Function(Output, Target_output)
%   This function is used to calculate the loss in a multi-label neural
% network, which is modified from common functions (e.g. square loss
% function) to adapt to features of multi-label learning, specifically,
% correlations between different labels. Detailed information on this 
% loss function is available in the original paper.
%   Reference:
% [1] M.Zhang, Z.Zhou. "Multilabel Neural Networks with Applications to 
%     Functional Genomics and Text Categorization," IEEE Transcations on 
%     Knowledge and Data Engineering, 2006.

% Notations are same to the orginal paper
% Online learning style is adopted.

% Initialize loss
loss = 0;

% Look for y and ~y(ny).
y = find(Target_output == 1);
ny = find(Target_output == -1);
% We make the assumption that each sample should have at least one label.
if(isempty(y)|| (size(y,1)+size(ny,1)) ~= size(Target_output,1))
    error('Wrong Target Output!');
end

% Calculate loss
for i = 1 : size(y, 1)
    for j = 1 : size(ny, 1)
        loss = loss + exp(-(Output(y(i,1),1) - Output(ny(j,1),1))); % Main body of the loss function
    end
end

% Normalize
loss = loss / size(y,1) / size(ny,1);

end

