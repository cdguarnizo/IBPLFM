function gParam = ibpmultigpLowerBoundGradients(model)

% IBPMULTIGPLOWERBOUNDGRADIENTS

% IBPMULTIGP

% MODIFICATIONS: Cristian Guarnizo, 2014

%[dLdKyy, dLdKuy, dLdKuu, dLdmu, gBeta] = spmultimodelLocalCovGradient2(model);
[dLdKyy, dLdKyu, dLdKuu, gBeta] = ibpmultigpLowerBoundGradCovMat(model);
fhandle = str2func([model.betaTransform 'Transform']);
gBeta = gBeta.*fhandle(model.beta, 'gradfact');

if isfield(model, 'gamma') && ~isempty(model.gamma)
    gGamma = zeros(1, model.nlf);
    for i =1:model.nlf
        gGamma(i) = trace(dLdKuu{i});
    end
    fhandle = str2func([model.gammaTransform 'Transform']);
    gGamma = gGamma.*fhandle(model.gamma, 'gradfact');
else
    gGamma = [];
end

% Modified by Mauricio Alvarez, May 08 2014
gKernParam = kernGradient(model.kern, model.outX, model.latX, dLdKyy, dLdKyu, dLdKuu);

if isfield(model, 'meanFunction') && ~isempty(model.meanFunction)
    if  strcmp(model.kernType,'lfm') || strcmp(model.kernType,'sim') ...
            || strcmp(model.kernType,'lfmwhite') ...
            || strcmp(model.kernType,'simwhite') ...
            || strcmp(model.kernType,'simnorm') ...
            || strcmp(model.kernType,'simglobal')
        gmu = zeros(1,model.nout);
        for j = 1:model.nout
            gmu(j) = sum(dLdmu{j});
        end
        g_meanFunc = meanGradient(model.meanFunction, gmu);
    else
        g_meanFunc = [];
    end
else
    g_meanFunc = [];
end

%Calculate derivative wrt adq and bdq
%Expensive way
% gbdqparams = zeros(model.nout,model.nlf);
% gadqparams = zeros(model.nout,model.nlf);
% for i = 1:model.nout,
%     for j = 1:model.nlf,
%         gbdqparams(i,j) = model.adq(i,j)/model.bdq(i,j) - model.adqast(i,j)/model.bdqast(i,j);
%         gadqparams(i,j) = -psi(model.adq(i,j)) + log(model.bdq(i,j)) + psi(model.adqast(i,j)) - log(model.bdqast(i,j));
%     end
% end

%Fast way
%gadqparams = psi(model.adqast) - psi(model.adq) - log(model.bdqast)...
%        + log(model.bdq) + 0.5*(model.EZdqS2dq - model.etadq).*psi(1,model.adqast)...
%        - 1./(2*model.bdqast).*model.EZdqS2dq - model.bdq./model.bdqast + 1;

if model.gammaPrior && isfield(model, 'adq') && ~isempty(model.adq)
    gadqparams = -psi(model.adq) + log(model.bdq) + psi(model.adqast) - log(model.bdqast);
    fhandle = str2func([model.adqTransform 'Transform']);
    gadqparams = gadqparams.*fhandle(model.adq, 'gradfact');
else
    gadqparams = [];
end

%gbdqparams = -(model.adq + 0.5* model.EZdqS2dq)./model.bdqast + (0.5*...
%    model.adqast.*model.EZdqS2dq + model.bdq.*model.adqast)./(model.bdqast.^2) ...
%    + model.adq./model.bdq - model.adq./model.bdqast; 

if model.gammaPrior && isfield(model, 'bdq') && ~isempty(model.bdq)
    gbdqparams = model.adq./model.bdq - model.adqast./model.bdqast;
    fhandle = str2func([model.bdqTransform 'Transform']);
    gbdqparams = gbdqparams.*fhandle(model.bdq, 'gradfact');
else
    gbdqparams = [];
end

if isfield(model, 'gradX') && model.gradX
    gX = ggglobalKernGradient_X(model.kern, model.outX, model.latX, dLdKyu, dLdKuu);
else
    gX = [];
end

gParam = [gKernParam g_meanFunc gGamma gBeta gadqparams(:)' gbdqparams(:)' gX];

if isfield(model, 'fix')
    for i = 1:length(model.fix)
        gParam(model.fix(i).index) = 0;
    end
end