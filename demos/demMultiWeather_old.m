% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel_old','../globalkern',genpath('../toolbox'),'../utils')

% load data
name = 'weather_old';
load ../datasets/weather/weatherdata.mat
D = size(y,1);

names = {'Bramblemet','Cambermet','Chimet','Sotonmet'};
test_ind{1} = xT{2} >= 10.2 & xT{2} <= 10.8;
test_ind{2} = xT{3} >= 13.5 & xT{3} <= 14.2;
outs = [2,3];
nout = length(outs);

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'gg';
options.optimiser = 'scg2';

options.nlf = 6;
options.numActive = 200;
options.alpha = 1;
options.beta = 1e-2;

options.NI = 200;
options.NIO = 20;

options.gamma = exp(-2);
options.initialInducingPositionMethod = 'espacedInRange';
options.Trainkern = true;
options.Trainvar = true;
options.isVarS = true;
options.nout = D;
options.kern.isArd = false;
options.kern.isVarS = options.isVarS;
options.gammaPrior = false;
options.gradX = false;
options.sparsePriorType = 'ibp';
options.asFinale = true;
options.pso = false;
options.actyhat = true;

%% Build and train the model
% Creates the model and initialize the hyperparameters
model = ibpmultigpCreate(x, y, options);
% Change initial conditions
model.beta = 1e3*ones(1, model.nout);

if ~options.Trainkern
    model.kern = lfmglobalKernExpandParam(model.kern, KernParams);
    model = ibpmultigpComputeKernels(model);
else
    [params, names] = ibpmultigpExtractParam(model);
    %index = paramNameRegularExpressionLookup(model, 'inverse width latent: .*');
    %params(index) = log(rand(1, length(index)));
    params(1:model.nlf) = log(1+0.2*randn(1, options.nlf));
    model = ibpmultigpExpandParam(model, params);
end
model.Trainvar = options.Trainvar;
if ~options.Trainvar
    model.etadq = Zdq;
    model.muSdq = Sdq;
    model.varSdq = zeros(size(Sdq));
end

NI = options.NI;
LB = zeros(10,NI);
K = zeros(10,1);
ZT = zeros(10,options.nout*options.nlf);
ST = zeros(10,options.nout*options.nlf);
rng(1e6)
Bestll = -inf;
for r = 1:10
    % Compute kernels
    model = ibpmultigpComputeKernels(model);
    % Initialize moments
    model = ibpmultigpMomentsInit(model);
    for k=1:NI
        % Perform E-Step
        % update variational dist. moments
        if model.Trainvar
            model = ibpmultigpMomentsCompute2(model);
        else
            model = ibpmultigpMomentEuast(model);
        end
        
        if (mod(k,10) == 0) && options.Trainkern %Every 10 iterations kern hyperpamaters are updated
            display = 0;
            iters = options.NIO;
            [model, opts, params] = ibpmultigpOptimise(model, display, iters);
        end
        
        if strcmp(model.sparsePriorType,'spikes')
            model.pi = sum(model.etadq(:))/(model.nout*model.nlf);
        end
        
        if ~model.gammaPrior
            model.gammadq = 1./(model.muSdq.^2 + model.varSdq);
        end
        LB(r,k) = ibpmultigpLowerBound2(model);
    end
    %temp = find(sum(abs(model.etadq.*model.muSdq)~=0));
    %K(r) = length(temp);
    K(r) = sum(sum(round(model.etadq))>=1);
    savemodel(model,r,name)
    ZT(r,:) = model.etadq(:)';
    ST(r,:) = model.muSdq(:)';
    if Bestll < max(LB(r,:))
        N = r;
        Bestll = max(LB(r,:));
    end
end

K2 = zeros(1,10);
for r = 1:10
    Ztemp = reshape(ZT(r,:), options.nout, options.nlf);
    Stemp = reshape(ST(r,:), options.nout, options.nlf);
    tempZ = find(sum(abs(Ztemp) >= 3e-1) >= 1);
    tempS = find(sum(abs(Stemp) >= 1e-2) >= 1);
    temp = intersect(tempZ, tempS);
    K2(r) = length(temp);
end

save(strcat('temp/',name),'LB','K','K2','ZT','ST','options','name','N')


%% Plot results from the selected solution
load(strcat('temp/',name,'.mat'));
load(strcat('temp/',name,num2str(N),'.mat'));

[ymean, yvar] = ibpmultigpPosterior(model, xT);

nmse = zeros(1,nout);
nlpd = zeros(1,nout);
fprintf('Best solution performance per output\n');
for k = 1:nout
    d = outs(k);
    ytest = yT{d}(test_ind{k});
    xtest = xT{d}(test_ind{k});
    
    nmse(k) = mysmse(ytest,ymean{d}(test_ind{k}));
    nlpd(k) = mynlpd(ytest,ymean{d}(test_ind{k}),yvar{d}(test_ind{k}));
    
    fprintf('Output %d, NMSE: %f, NLPD: %f\n',outs(k),nmse(k),nlpd(k));
    
    %Plot outputs
    figure(k)
    plotGP(ymean{d}, yvar{d}, xT{d}, y{d}, x{d}, ytest, xtest);
    %title(names{d})
end
%Plot hinton diagram
hinton(model.kern.sensitivity(:,1:K(N)).*model.etadq(:,1:K(N)));