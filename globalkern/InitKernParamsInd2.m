function [KernOutParams S] = InitKernParamsInd2(X, y, kernType, nlf)
% Covariance parameters for each output is calculeted individually

% Set the Options
options = gpOptions('ftc');
options.kern = {kernType, 'white'};
options.scale2var1 = 1;
itersSingleGp = 200;
invwidthini = [0.1 1 10 100];

if strcmp(kernType,'gg'),
    %Variance
    options.fix(1).index = 3;
    options.fix(1).value = expTransform(1., 'xtoa');
elseif strcmp(kernType,'sim'),
    %Variance
    options.fix(1).index = 3;
    options.fix(1).value = expTransform(1., 'xtoa');
end

D = max(size(y));
ndim = size(X{1},2);

KernOutParams = cell(1, D);
S = cell(1, D);
% Creates the model
mset = inf;
for d = 1:D,
    for initial=1:length(invwidthini), %Test different lengthscales initializations
        model = gpCreate(ndim, 1, X{d}, y{d}, options);
        index = paramNameRegularExpressionLookup(model, '.* inverse .*');
        params = gpExtractParam(model);
        params(index) = log(invwidthini(initial) + 0.2*invwidthini(initial)*rand(1,length(index)));
        model = gpExpandParam(model, params);
        
        model = gpOptimise(model, 1, itersSingleGp);
        [ymean ~] = gpPosteriorMeanVar(model, X{d});
        mse = mean((y{d} - ymean).^2);
        if mse < mset,
            mset = mse;
            [params ~] = gpExtractParam(model);
        end
    end
    [KernOutParams{d} S{d}] = ExtractGPParams(params, kernType);
end
KernOutParams = cell2mat(KernOutParams);
KernOutParams = KernOutParams(:)';
ls = linspace(min(KernOutParams(1:D)),max(KernOutParams(1:D)),nlf);
%ls = mean(KernOutParams(1:D)) + 0.2*mean(KernOutParams(1:D))*rand(1,nlf);
KernOutParams = [ls KernOutParams(D+1:end)];
S = cell2mat(S);
S = S(:);