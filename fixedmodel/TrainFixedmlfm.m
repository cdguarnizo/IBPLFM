function [model, Fold] = TrainFixedmlfm(y, x, options)

addpath(genpath('../toolbox'),'../globalkern','../utils','../ftcmgp');

options.gamma = exp(-2);
options.kern.isVarS = options.isVarS;
options.kern.isArd = false;
options.nout = size(y,1);
ndim = size(x{1},2);

if ndim>1,
    options.initialInducingPositionMethod = 'kmeansHeterotopic';
else
    options.initialInducingPositionMethod = 'espacedInRange';
end

if options.nlf > 1,
    warning('off','multiKernParamInit:noCrossKernel');
end

model = ibpmultigpCreate(x, y, options);

model.Trainkern = true;
model.Trainvar = true;
model = ibpmultigpComputeKernels(model);
model = ibpmultigpMomentsInit(model);


%% Initialization of kernel and variational parameters
if isfield(options,'InitInvWidth'),
    [params, ~] = ibpmultigpExtractParam(model);
    if length(options.InitInvWidth)==1,
        params(1:model.nlf) = log(options.InitInvWidth + 0.1*options.InitInvWidth*randn(1,model.nlf));
    else
        params(1:model.nlf) = log(options.InitInvWidth);
    end
    model = ibpmultigpExpandParam(model, params);
else
    [params, ~] = ibpmultigpExtractParam(model);
    params(1:model.nlf) = log(1 + .1*randn(1,model.nlf));
    model = ibpmultigpExpandParam(model, params);
end

model.etadq = model.Z;
model.isVarU = false;

[model, ~, ~] = ibpmultigpOptimise(model, options.DispOpt, 500);
[model.Euast, model.Kuuast, model.logDetKuuast, model.Euuast] = ibpmultigpUpdateLatent(model,0);


Fold = ibpmultigpLowerBound(model);