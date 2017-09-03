function [model, Fold, mae, mse, msmse , mmsll] = TrainSparseMGP3(y, x, yTest, xTest, options)

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

%Normalize data
if ndim > 1,
    for d=1:options.nout,
        options.bias(d) = mean(y{d});
        options.scale(d) = std(y{d});
    end
else
    for d=1:options.nout,
        options.bias(d) = y{d}(1);
        %options.bias(d) = mean(y{d});
        options.scale(d) = std(y{d});
        %options.scale(d) = 1;
    end
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

% if options.InitKern,
%     opt = options;
%     opt.approx = 'ftc';
%     %opt.nlf = 1;
%     opt.isVarS = false;
%     opt.beta = 1e-2;
%     modelT = ftcmgpCreate(cell2ind(xTemp), cell2mat(yTemp), opt);
%     modelT = ftcmgpOptimise(modelT, 2 , 100);
%     modelT.kern.options.isVarS = true;
%
%     KernOutParams = kernExtractParam(modelT.kern);
%     model.kern = kernExpandParam(model.kern, KernOutParams);
%     if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp'),
%         model.muSdq = modelT.kern.sensitivity;
%         model.varSdq = ones(size(model.muSdq));
%     end
%     model.beta = modelT.beta;
%     clear modelT opt KernOutParams
%     model = ibpmultigpComputeKernels(model);
%     for k=1:10,
%         model = ibpmultigpMomentEuast(model);
%     end
% end

% [KernOutParams S] = InitKernParamsInd2(xTemp, yTemp, model.kernType(1:end-6), model.nlf);
% model.kern = kernExpandParam(model.kern, KernOutParams);

% if isfield(options,'InitTrainKern') && options.InitTrainKern,
%     etadq = model.etadq;
%     isVarS = model.isVarS;
%     model.isVarS = 0;
%     model.etadq(:) = 1.;Iteration: 20, LB: -315.004376, Increment: -304.652607

%     [model, ~, ~] = ibpmultigpOptimise(model, 1, 100);
%     model.etadq = etadq;
%     model.isVarS = isVarS;
%     if model.isVarS,
%         model.muSdq = model.kern.sensitivity;
%     end
%     clear etadq isVarS
% end

if options.InitKern,
    fprintf('Initializing kern model.\n')
    %tempeta = model.etadq;
    %model.etadq = ones(size(model.etadq));
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
    
    [model, ~, ~] = ibpmultigpOptimise(model, 2, 100);
    
    if options.OptMarU,
        if options.isVarU, 
            [model.Euast, model.Kuuast] = ibpmultigpUpdateLatent(model);
        end
        model.isVarU = options.isVarU;
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
Fold = zeros(1,options.NI+1);
Fold(1) = ibpmultigpLowerBound(model);
fprintf('Iteration: 0, LB: %f\n',Fold(1));
for k = 1:options.NI,
    % Update variational dist. moments
    model = ibpmultigpMomentsCompute2(model);

    if ~model.Opteta,
        % Update pi value if sparse prior is spike and slab
        if strcmp(model.sparsePriorType,'spikes'),
            model.pi = sum(model.etadq(:))/(model.nout*model.nlf);
        end
        
        %Optimize gammadq if model.gammaPrior is false
        if ~model.gammaPrior && model.isVarS,
            model.gammadq = 1./(model.muSdq.^2 + model.varSdq);
            %model.gammadq(model.gammadq == Inf) = 1e6;%TODO: check this hack
        end
    end
    
    %Optimize hyperparameters
    if mod(k,10)==0 && model.Trainkern,
        
        %Sort model according to eta values
        if options.sorteta,
            model = ibpmultigpSortModel(model);
        end

        model.etadq
        
        if model.debug,
            F1 = ibpmultigpLowerBound(model);
        end
        
        if options.OptMarU,
            model.isVarU = false;
        end
        %params = ibpmultigpExtractParam(model);
        %try
            [model, ~, ~] = ibpmultigpOptimise(model, options.DispOpt, options.NIO);
        %catch err,
        %    model = ibpmultigpExpandParam(model, params);
        %end
        if options.OptMarU,
            model.isVarU = options.isVarU;
            if model.isVarU,
                
%                 F1 = ibpmultigpLowerBound(model);
%                 delta = 1e-2;
%                 con = 0;
%                 while delta > 1e-3,
%                     con = con + 1;
%                     model = ibpmultigpMomentEuast(model);
%                     F2 = ibpmultigpLowerBound(model);
%                     delta = F2 - F1;
%                     %fprintf('After updating opt.: %f\n', delta)
%                     F1 = F2;
%                     if con == 10,
%                         delta = 1e-4;
%                     end
%                 end
                
                [model.Euast, model.Kuuast] = ibpmultigpUpdateLatent(model);
            end
        end
        
        % Set a maximum for noise precisions
        %model.beta(model.beta > 1e4) = 1e4;
        if model.debug,
            F2 = ibpmultigpLowerBound(model);
            fprintf('Optimization: %d, LB: %f, Increment: %f\n',k,F2, F2-F1);
        end
    end
    Fold(k+1) = ibpmultigpLowerBound(model);
    fprintf('Iteration: %d, LB: %f, Increment: %f\n',k,Fold(k+1), Fold(k+1)-Fold(k));
end

[ymean, yvar] = ibpmultigpPosterior(model, xTest);
[mae, mse, smse, msll, ~] = multigpErrorMeasures(y, yTest, ymean, ...
    yvar, model.nout);
msmse = mean(smse);
mmsll = mean(msll);