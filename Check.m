function [boolean] = Check(x, y, type)
% type = 0: check if x = y;
% type = 1: check if x > y;
% type = 2: check if x < y;
% type = others: undefined;

if(type == 0)
    boolean = ~(x == y);
elseif(type == 1)
    boolean = ~(x > y);
elseif(type == 2)
    boolean = ~(x < y);
else
    error('Undefined Check Type!');
end

end

