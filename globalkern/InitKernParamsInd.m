function [KernOutParams S] = InitKernParamsInd(X, y, kernType)
% Covariance parameters for each output is calculeted individually

% Set the Options
options = multigpOptions('ftc'); % DTCVAR?
options.kernType = kernType;
options.optimiser = 'scg';
options.nlf = 1;
options.beta = 1e3;
options.gamma = 1e-1;
options.includeNoise = false;
D = max(size(y));
ndim = size(X{1},2);

options.numActive = 8;
options.initialInducingPositionMethod = 'espacedInRange';
options.fixInducing = true;

warning('off', 'multigpCreate:NoTyingFunctionGlobalKernel')
warning('off', 'multiKernParamInit:noCrossKernel')

KernOutParams = cell(1, D);
S = cell(1, D);
% Creates the model
for d = 1:D,
    model = multigpCreate(ndim, 1, {X{d}}, {y{d}}, options);
    
    % Set parameters associated with the inverse widths to different values for
    % symmetry breaking.
    
    if options.nlf>1,
        params = modelExtractParam(model);
        for i = 1:options.nlf
            paramInd = paramNameRegularExpressionLookup(model, ['multi ' num2str(i) ...
                ' .* inverse width']);
            params(paramInd) = params(paramInd) + 0.1*randn;
        end
        model = modelExpandParam(model, params);
    end
    display = 0;
    iters = 200;
    
    model = multigpOptimise(model, display, iters);
    [params ~] = multigpExtractParam(model);
    [KernOutParams{d} S{d}] = ExtractGPmatParams(params, kernType, 1, options.nlf);
end
KernOutParams = cell2mat(KernOutParams);
KernOutParams = KernOutParams(:);
S = cell2mat(S);
S = S(:);