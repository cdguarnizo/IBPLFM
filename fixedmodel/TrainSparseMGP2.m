function [model, Fold] = TrainSparseMGP2(y, x, options)

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

if options.InitSearchS,
    model.muSdq = reshape(pso(model),model.nout, model.nlf);
end

if options.InitKern,
    fprintf('Initializing kern model.\n')
    tempeta = model.etadq;
    model.etadq = ones(size(model.etadq));
    OptMarU = options.OptMarU;
    options.OptMarU = true;
    if model.isVarS,
        model.isVarS = false;
        model.kern.isVarS = false;
        model.kern.options.isVarS = false;
        SnParam = model.nlf*model.nout;
        model.nParams = model.nParams + SnParam;
        model.kern.nParams = model.kern.nParams + SnParam;
    end
    if options.OptMarU && options.isVarU,
       model.isVarU = false;
    end
    
    [model, ~, ~] = ibpmultigpOptimise(model, options.DispOpt, 50);
    
    if options.OptMarU,
        model.isVarU = options.isVarU;
        if options.isVarU,
            [model.Euast, model.Kuuast, model.logDetKuuast, model.Euuast] = ibpmultigpUpdateLatent(model,0);
        end
    end
    
    KernOutParams = kernExtractParam(model.kern);
    model.kern = kernExpandParam(model.kern, KernOutParams);
    if options.isVarS,
        model.isVarS = options.isVarS;
        model.kern.isVarS = options.isVarS;
        model.kern.options.isVarS = options.isVarS;
        model.muSdq = model.kern.sensitivity;
        model.kern.sensitivity = ones(size(model.muSdq));
        model.varSdq = ones(size(model.muSdq));
        model.nParams = model.nParams - SnParam;
        model.kern.nParams = model.kern.nParams - SnParam;
    end
    options.OptMarU = OptMarU;
    model.etadq = tempeta;
end

fprintf('Performing variational inference\n')
Fold = zeros(1,options.NI);
%Fold(1) = ibpmultigpLowerBound(model);
%fprintf('Iteration: 0, LB: %f\n',Fold(1));
for k = 1:options.NI,
    % Update variational dist. moments
    model = ibpmultigpMomentsCompute(model);

    if ~model.Opteta,
        % Update pi value if sparse prior is spike and slab
        if strcmp(model.sparsePriorType,'spikes'),
            model.pi = sum(model.etadq(:))/(model.nout*model.nlf);
        end
        
        %Optimize gammadq if model.gammaPrior is false
        if ~model.gammaPrior && model.isVarS,
            model.gammadq = 1./(model.muSdq.^2 + model.varSdq);
        end
    end

    %Optimize hyperparameters
    if mod(k,10)==0 && model.Trainkern && k~=options.NI,
        
        %Find correlated latent functions
        if mod(k,50)==0. &&  k~=options.NI,
            [model.etadq, model.kern.sensitivity] = ibpmultigpPruning(model.Euast,...
                model.etadq, model.kern.sensitivity);
        end
        
        %Sort model according to eta values
        if options.sorteta,
            model = ibpmultigpSortModel(model);
        end
        
        if model.debug,
            F1 = ibpmultigpLowerBound(model);
        end
        
        if options.OptMarU,
            model.isVarU = false;
        end
        [model, ~, ~] = ibpmultigpOptimise(model, options.DispOpt, options.NIO);
        %Delete expensive kernel
        model.Kfu = [];
        if options.OptMarU,
            model.isVarU = options.isVarU;
            if model.isVarU,
                [model.Euast, model.Kuuast, model.logDetKuuast, model.Euuast] = ibpmultigpUpdateLatent(model,0);
            end
        end
        % Set a maximum for noise precisions
        %model.beta(model.beta > 1e4) = 1e4;
        if model.debug,
            F2 = ibpmultigpLowerBound(model);
            fprintf('Optimization: %d, LB: %f, Increment: %f\n',k,F2, F2-F1);
        end
    end
    Fold(k) = ibpmultigpLowerBound(model);
    if k>1,
        fprintf('Iteration: %d, LB: %f, Increment: %f\n',k,Fold(k), Fold(k)-Fold(k-1));
    end
end