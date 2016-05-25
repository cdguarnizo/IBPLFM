function Xind = cell2ind(X)
% This funcion allow to convert from cell to index scheme

Xind = struct('ind',[],'val',[]);
Xind.val = cell2mat(X);
Xind.ind = zeros(size(Xind.val,1),1);
endVal = 0;
for d = 1:size(X,1),
    startVal = endVal + 1;
    endVal = startVal + size(X{d},1) - 1;
    Xind.ind(startVal:endVal) = d;
end