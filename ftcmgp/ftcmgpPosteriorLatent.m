function [meanval, varval] = ftcmultigpPosteriorLatent(model, Xtest)

% FTCMULTIGPPOSTERIOR
% FTCMULTIGP

fhandle = str2func([model.kernType,'XrbfKernCompute']);
Kfu = cell2mat(fhandle(model.kern, model.outX, Xtest));

meanval = Kfu'*model.alpha;
if nargout>1,
    dimenu = length(cell2mat(Xtest));
    Kuu = zeros(dimenuy, dimenu);
    start = 1;
    for q = 1:model.nlf,
        temp = Xtest{q}/kern.lq(q);
        ind = start: start + length(Xtest{q}) - 1;
        Kuu(ind, ind) = exp(-dist2(temp, temp));
        start = ind(end)+1;
    end
    varval = Kuu - Kfu'*model.Kyyinv*Kfu;
end