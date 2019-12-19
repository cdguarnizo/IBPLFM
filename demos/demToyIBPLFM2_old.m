% DEMO based on Toy Data using the old inference model

% MULTIGP

clc
clear
close all
format short e
%s = RandStream('mt19937ar', 'Seed', 3e4); %5e5 8e5 9e5
%RandStream.setGlobalStream(s);
addpath('../sparsemodel_old','../globalkern',genpath('../toolbox'),'../utils')

% load data
name = 'IBPLFM_old';
load ../datasets/Toys/datasetD3Q2_IBP_LFM2_new_noise.mat

names = {'Output 1','Output 2','Output 3'};

outs = [1,2];
nout = length(outs);
test_ind = cell(length(outs),1);
test_ind{1} = 32:45;
test_ind{2} = 21:30;

D = size(fd,2);
y = cell(D,1);
yT = y;
x = cell(D,1);
xT = x;
for d = 1:D
    y{d} = fd(:,d);
    x{d} = xTemp;
    yT{d} = fd(:,d);
    xT{d} = xTemp;
    if any(d == outs)
        ind = find(d==outs);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end
clear fd xTemp


%% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'lfm';
options.optimiser = 'scg2';

options.nlf = 4;
options.numActive = 15;
options.alpha = 1;
options.beta = 1;

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
    options.bias(d) = 0;%yT{d}(1);
    options.scale(d) = 1;%std(yT{d});
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

NI = 300;
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
            iters = 100;
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

save Toy_old LB K K2 ZT ST Zdq Sdq options name N


%%
load('Toy_old.mat')
load(strcat('temp/',name,num2str(N)));

close all
[ymean, yvar]=ibpmultigpPosterior(model, xT);

nmse = zeros(1,nout);
nlpd = zeros(1,nout);
for d = 1:D
    if any(outs == d)
        ytest = yT{d}(test_ind{d});
        xtest = xT{d}(test_ind{d});
        nmse(d) = mysmse(ytest,ymean{d}(test_ind{d}));
        nlpd(d) = mynlpd(ytest,ymean{d}(test_ind{d}),yvar{d}(test_ind{d}));
    else
        ytest = [];
        xtest = [];
    end
    %plot(xT{d},ymean{d},'k');
    %hold on
    %plot(model.outX{d},model.y{d},'.k');
    %plot(model.latX{1},zeros(options.numActive,1),'x')
    figure(d)
    plotGP(ymean{d}, yvar{d}, xT{d}, y{d}, x{d}, ytest, xtest);
end
%plotpred2(model, model.outX, model.y)

[umean, uvar] = ibpmultigpPosteriorLatent(model, xT{1});
figure(options.nout+1)
%plot(xT{1},uq(:,1)/max(uq(:,1)),xT{1},umean{3}/max(umean{3}))
plotGP(umean{3}*model.muSdq(1,3), diag(uvar{3})*model.muSdq(1,3)^2, xT{1})
figure(options.nout+2)
%plot(xT{1},uq(:,2)/max(uq(:,2)),xT{1},umean{1}/max(umean{1}))
plotGP(umean{1}*model.muSdq(1,1), diag(uvar{1})*model.muSdq(1,1)^2, xT{1})


hinton(model.etadq(:,[3,1]).*model.muSdq(:,[3,1]))
%Comparison of hyperparameters
%estKernParams = lfmglobalKernExtractParam(model.kern);

%%
for k=1:10
    hinton(reshape(ZT(k,:).*ST(k,:),3,4))
end