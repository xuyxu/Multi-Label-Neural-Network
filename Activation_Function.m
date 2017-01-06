function [output] = Activation_Function(input, method)
%   This function serves as the activation function in multi-label neural
% network. Specifically, Sigmod, tanh, and Relu functions are supported.
% Other activation functions are also available with additional codes.
if(strcmp(method,'sigmoid') || strcmp(method,'Sigmoid'))
    
    output = 1 ./ (1 + exp(-input));
   
elseif(strcmp(method,'tanh') || strcmp(method,'Tanh'))
    
    output = (exp(input) - exp(-input)) ./ (exp(input) + exp(-input));
    
elseif(strcmp(method, 'relu') || strcmp(method,'Relu'))
    
    output = max(0,input);

    % Other activation functions can be added here, similar to the form
    % above

else
    error('Undesignated activation functions! Please check the input of function Activation_Function()');
end

end

