function [KernOutParams S] = InitKernParams(X, y, kernType, nlf)
% Initialization of covarianve paramters by training the whole dataset using
% DTCVAR.

% Set the Optionsmse, mslls
options = multigpOptions('dtcvar');
options.kernType = kernType;
options.optimiser = 'scg';
options.nlf = nlf;
options.numActive = 10;
options.beta = 1e3;
options.gamma = 1e-1;
options.fixInducing = true;
options.includeNoise = false;
D = max(size(y));
ndim = size(X{1},2);

if size(X{1},2)>1,
    options.initialInducingPositionMethod = 'kmeansHeterotopic';
else
    options.initialInducingPositionMethod = 'espacedInRange';
end

warning('off', 'multigpCreate:NoTyingFunctionGlobalKernel')
warning('off', 'multiKernParamInit:noCrossKernel')

% Creates the model
model = multigpCreate(ndim, D, X, y, options);

% Set parameters associated with the inverse widths to different values for
% symmetry breaking.

if options.nlf>1,
    params = modelExtractParam(model);
    for i = 1:options.nlf
        paramInd = paramNameRegularExpressionLookup(model, ['multi ' num2str(i) ...
            ' .* inverse width']);
        params(paramInd) = params(paramInd) + .1*randn;
    end
    model = modelExpandParam(model, params);
end
display = 0;
iters = 200;

% Trains the model and counts the training time
model = multigpOptimise(model, display, iters);

% [mu, varsigma] = multigpPosteriorMeanVar(model,  xTemp2);
% 
% for d=1:D,
%     figure(d);
%     plot([yTemp2{d} mu{d+options.nlf}])
% end
% [mae, mse, smse, msll] = multigpErrorMeasures(yTemp2, yTemp2, mu(model.nlf+1:end), ...
%     varsigma(model.nlf+1:end), model.nout);
% msmse = mean(smse);
% mmsll = mean(msll);

% Parameters extraxction from GPmat model
[params names] = multigpExtractParam(model);

if strcmp(kernType,'lfm'),
    np=5; %lengthscales mass spring dampes
    KernOutParams = [params(1:np*D+2:options.nlf*(np*D+1)+1) ...
        params(3:np:np*(D-1)+3) params(4:np:np*(D-1)+4) params(5:np:np*(D-1)+5)];
elseif strcmp(kernType,'sim'),
    np=3; %lengthscales variance decay
    KernOutParams = [params(1:np*D+2:options.nlf*(np*D+1)+1) ...
        params(3:np:np*(D-1)+3)];
else %GG kernel
    np=4; %lengthscales precU precG
    KernOutParams = [params(1:np*D+2:options.nlf*(np*D+1)+1) ...
        params(4:np:np*(D-1)+4)];
end

%Extract sensitivities
S = zeros(D, options.nlf);
start = 2 + np;
for q = 1:options.nlf,
    S(:,q) = params(start:np:start+(D-1)*np);
    start = start+(D-1)*np + (2 + np);
end
