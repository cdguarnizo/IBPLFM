function Xcell = ind2cell(X, ind, D)
% This function allows to convert values index scheme to cell
if nargin < 3,
    D = max(ind);
end

Xcell = cell(D,1);
for d = 1:D,
    Xcell{d} = X(ind == d);
end