function gParam = ftcmgpLowerBoundGradients(model)

% FTCMGPLOWERBOUNDGRADIENTS

% Cristian Guarnizo, 2015


% gParam = zeros(1, model.nParams);
% paramt = modelExtractParam(model);
% for k = 1:model.nParams,
%     p = paramt(k);
%     paramt(k) = p + 1e-6;
%     model = modelExpandParam(model, paramt);
%     model = ftcmgpComputeKernel(model);
%     gParam(k) = ftcmgpLowerBound(model);
%     
%     paramt(k) = p - 1e-6;
%     model = modelExpandParam(model, paramt);
%     model = ftcmgpComputeKernel(model);
%     gParam(k) = (gParam(k) - ftcmgpLowerBound(model))/2e-6;
%     
%     paramt(k) = p;
%     model = modelExpandParam(model, paramt);
% end


dLdKyy = model.alpha*model.alpha' - model.Kyyinv;

dKyy = cell(model.nout, model.nout);
for d = 1:model.nout,
    indd = model.outX.ind == d;
    dKyy{d,d} = dLdKyy(indd, indd);
    for dp = d+1:model.nout,
        inddp = model.outX.ind == dp;
        dKyy{d,dp} = dLdKyy(indd, inddp);
        dKyy{dp,d} = dLdKyy(inddp, indd);
    end
end

gBeta = [];
if model.includeNoise && isfield(model, 'beta'),
    gBeta = zeros(1,model.nout);
    temp = diag(dLdKyy);
    for d = 1:model.nout,
        gBeta(d) = (-1/model.beta(d)^2)*sum(temp(model.outX.ind == d));
    end
    fhandle = str2func([model.betaTransform 'Transform']);
    gBeta = .5*gBeta.*fhandle(model.beta, 'gradfact');
end

gKernParam = kernGradient(model.kern, ind2cell(model.outX.val, model.outX.ind), [], dKyy);

gParam = [gKernParam gBeta];
 
if isfield(model, 'fix'),
    for i = 1:length(model.fix),
        gParam(model.fix(i).index) = 0;
    end
end