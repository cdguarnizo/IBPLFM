% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel_old','../globalkern',genpath('../toolbox'),'../utils')

% load data
fd = amc_to_matrix('../datasets/CMUmocap/02_03.amc');

% 8 signals test
names = {'rhumerus','rradius','rfemur','lhumerus','lradius','lfemur'};
IndSel = [27,30,49,39,42,56];
name = 'MOCAP_old';
outs = [1,4];
nout = length(outs);
fd = fd(:,IndSel);
[N, D] = size(fd);
t = (1:N)'/120; %Time stamp in seconds

%Downsample
IndDown = 1:4:N;

test_ind{1} = t(IndDown) >= .9 & t(IndDown) <= 1.2;
test_ind{2} = t(IndDown) >= .35 & t(IndDown) <= .65;

y = cell(D,1);
x = cell(D,1);
xT = cell(D,1);
yT = cell(D,1);
for d = 1:D
    y{d} = fd(IndDown, d);
    x{d} = t(IndDown);
    yT{d} = y{d};
    xT{d} = x{d};
    if any(d == outs)
        ind = find(d==outs);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end

clear fd t

% Set model Options 
options = ibpmultigpOptions('dtcvar');
options.sparsePriorType = 'ibp';
options.kernType = 'lfm';
options.optimiser = 'scg2';

options.nlf = 6;
options.numActive = 25;
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
options.nout = D;
options.kern.isArd = false;
options.kern.isVarS = options.isVarS;
options.gammaPrior = false;
options.gradX = false;
options.sparsePriorType = 'ibp';
options.asFinale = true;
options.pso = false;
options.actyhat = true;

for d = 1:D
    options.bias(d) = yT{d}(1);
    options.scale(d) = std(yT{d});
end

%% Training IBPLFM variational approach
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

NI = 200;
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
            iters = 20;
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
    temp = find(sum(abs(model.etadq.*model.muSdq)~=0));
    savemodel(model,r,name)
    ZT(r,:) = model.etadq(:)';
    ST(r,:) = model.muSdq(:)';
    K(r) = length(temp);
    if Bestll < max(LB(r,:))
        N = r;
        Bestll = max(LB(r,:));
    end
end

K2 = zeros(1,10);
for r = 1:10
    Ztemp = reshape(ZT(r,:), options.nout, options.nlf);
    Stemp = reshape(ST(r,:), options.nout, options.nlf);
    tempZ = find(sum(abs(Ztemp) >= 1e-1) >= 1);
    tempS = find(sum(abs(Stemp) >= 1e-2) >= 1);
    temp = intersect(tempZ, tempS);
    K2(r) = length(temp);
end

save MOCAP_old LB K K2 ZT ST options N


%% Plot results from the selected solution
close all
load('MOCAP_old.mat')
load(strcat('temp/',name,num2str(N)));

[ymean, yvar, ~] = ibpmultigpPosterior(model, xT);

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
hinton(model.kern.sensitivity.*model.etadq);