function [model, Fold, msmse , mmsll] = TrainSparseMGP(y, x, options)

addpath(genpath('../toolbox'),'../globalkern','../utils','../ftcmgp');

options.beta = 1e-3;
options.gamma = exp(-2);
options.isVarS = true;
options.kern.isVarS = options.isVarS;
options.kern.isArd = false;

if size(x{1},2)>1
    options.initialInducingPositionMethod = 'kmeansHeterotopic';
else
    options.initialInducingPositionMethod = 'espacedInRange';
end

if ~iscell(y)
    options.nout = size(y, 2);
    yTemp = cell(1,options.nout);
    xTemp = cell(1,options.nout);
    yTemptest = cell(1,options.nout);
    xTemptest = cell(1,options.nout);
    % Output normalization for convolved Gaussian Processes
    for d = 1:options.nout
        scale = std(y(:,d));
        yTemp{d} = (y(:,d) - y(1,d))/scale;
        yTemptest{d} = (yTest(:,d) - y(1,d))/scale;
        xTemp{d} = x;
        xTemptest{d} = xTest;
    end
else
    options.nout = max(size(y));
    yTemp = y;
    xTemp = x;
    yTemptest = yTest;
    xTemptest = xTest;
end
clear y x yTest Xtest

% for i=1:d,
%     options.bias(i) = mean(yTemp2{i});
%     options.scale(i) = std(yTemp2{i});
% end

if options.nlf > 1
   warning('off','multiKernParamInit:noCrossKernel');
end

% Create and intilize the sparse model
model = ibpmultigpCreate(xTemp, yTemp, options);

model.Trainkern = true;
model.Trainvar = true;

if ~model.Trainkern
    model.kern = KernExpandParam(model.kern, KernParams);
    model = ibpmultigpComputeKernels(model);
else
    [params, names] = ibpmultigpExtractParam(model);
    %index = paramNameRegularExpressionLookup(model, 'inverse width latent: .*');
    %params(index) = log(1 + 0.2*randn(1, length(index)));
    params(1:model.nlf) = log(1+.2*randn(1,model.nlf));
    model = ibpmultigpExpandParam(model, params);
end

% Compute kernels
model = ibpmultigpComputeKernels(model);
% Initialize moments
model = ibpmultigpMomentsInit(model);

NI = options.NI; %Number of iterations for EM steps
paramsOld = ibpmultigpExtractParam(model);
Fold = ibpmultigpLowerBound2(model);
try
    for k=1:NI
        % Update variational dist. moments
        model = ibpmultigpMomentsCompute2(model);
        % Update pi value if sparse prior is spike and slab
        if strcmp(model.sparsePriorType,'spikes')
            model.pi = sum(model.etadq(:))/(model.nout*model.nlf);
        end
        
        % Optimize of gammadq if model.gammaPrior is false
        if ~model.gammaPrior
            if (strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'))
                model.gammadq = 1./(model.etadq.*(model.muSdq.^2 + model.varSdq));
            else
                model.gammadq = 1./(model.muSdq.^2 + model.varSdq);
            end
        end
        
        %Optimize hyperparameters
        if mod(k,10)==0 && model.Trainkern
            paramsOld = ibpmultigpExtractParam(model);
            display = 0; %Do not display gradient check
            iters = 100;
            [model, ~, ~] = ibpmultigpOptimise(model, display, iters);
            Fnew = ibpmultigpLowerBound2(model);
            model.beta
            if Fnew > Fold
                paramsOld = ibpmultigpExtractParam(model);
                Fold = Fnew
            end
        end
    end
catch ME
    model = ibpmultigpExpandParam(model, paramsOld);
end

% [ymean yvar]=ibpmultigpPosterior(model, xTemptest);
% 
% [mae, mse, smse, msll] = multigpErrorMeasures(yTemp, yTemptest, ymean, ...
%     yvar, model.nout);
% msmse = mean(smse);
% mmsll = mean(msll);