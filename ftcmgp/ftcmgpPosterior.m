function [meanval, varval] = ftcmgpPosterior(model, Xtest)

% FTCMULTIGPPOSTERIOR
% FTCMULTIGP

fhandle = str2func([model.kernType 'KernCompute']);
Kffsq = fhandle(model.kern, ind2cell(model.outX.val, model.outX.ind), ...
    ind2cell(Xtest.val, Xtest.ind));

Kffs = zeros(size(Kffsq{1}));

if isfield(model,'isVarS') && model.isVarS,
    for k = 1:model.nlf,
        SS = model.S(:, k)*model.S(:, k)';
        Kffs = Kffs + SS(model.outX.ind, Xtest.ind).*Kffsq{k};
    end
else
    for k = 1:model.nlf,
        Kffs = Kffs + Kffsq{k};
    end
end

meanval = Kffs'*model.alpha;
if nargout>1,
    Kfsfs = fhandle(model.kern, ind2cell(Xtest.val, Xtest.ind));
    Kfsfs = jitChol(Kfsfs);
    Kfsfs = Kfsfs*Kfsfs';
    varval = Kfsfs - Kffs'*model.Kyyinv*Kffs;
end