% DEMTOYIBPSIM Variational LFM with IBP prior over latent forces over the LFM on Yeast Spellman Data 
% DESC
% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel_old','../globalkern',genpath('../toolbox'),'../utils')

% Yeast data from TU
name = 'yeast_old';
load ../datasets/Yeast/Tudata_G8.mat
names = {'YLR183C','YLR030W','TKL2','YOR359W','PFK27','RPL17B','RPS16B', ...
    'RPL13A'}; 
t = t./350;

test_ind{1} = t(1,:) >= t(1,9) & t(1,:) <= t(1,17);
test_ind{2} = t(5,:) >= t(5,16) & t(5,:) <= t(5,25);

outs = [1,5];
nout = length(outs);
D = size(f,1);
y = cell(D,1);
x = cell(D,1);
xT = cell(D,1);
yT = cell(D,1);
for d = 1:D
    y{d} = f(d,:)';
    x{d} = t(d,:)';
    yT{d} = y{d};
    xT{d} = x{d};
    if any(d == outs)
        ind = find(d==outs);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'sim';
options.optimiser = 'scg2';

options.nlf = 8;
options.nout = D;
options.numActive = 18;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 0;
options.beta = 1e-2;

options.gamma = exp(-2);
options.initialInducingPositionMethod = 'espacedInRange';
options.Trainkern = true;
options.Trainvar = true;
options.isVarS = true;
options.kern.isArd = false;
options.kern.isVarS = options.isVarS;
options.gammaPrior = false;
options.gradX = false;
options.sparsePriorType = 'ibp';
options.asFinale = true;
options.pso = false;
options.actyhat = true;

for d = 1:D
    options.bias(d) = mean(yT{d});
    options.scale(d) = std(yT{d});
end

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
    title(names{d})
end
%Plot hinton diagram
hinton(model.kern.sensitivity.*model.etadq);