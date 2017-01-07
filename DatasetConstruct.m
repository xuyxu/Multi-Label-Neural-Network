function [example] = DatasetConstruct(xid, Alldata_, y1, y2)

example.x = zeros(size(Alldata_,1),1);
temp = zeros(size(xid,1),1);

for i = 1 : size(xid, 1)
    for j = 1 : size(Alldata_, 1)
        if(strcmp(xid{i,1},Alldata_{j,1}.ID))
            temp(i,1) = j;
        end
    end
end

for i = 1 : size(temp,1)
    example.x(temp(i,1),1) = 1;
end

example.y = [y1, y2];
            
end

