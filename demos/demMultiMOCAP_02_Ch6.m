% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')

% load data
fd = amc_to_matrix('../datasets/CMUmocap/02_03.amc');

% 8 signals test
names = {'rhumerus','rradius','rfemur','lhumerus','lradius','lfemur'};
IndSel = [27,30,49,39,42,56];
name = 'MOCAP';
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
        ind = find(outs==d);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end

clear fd t

% Set model Options 
options = ibpmultigpOptions('dtcvar');
options.sparsePriorType = 'ibp';
options.kernType = 'lfm';
options.optimiser = 'scg';

options.nlf = 6;
options.numActive = 25;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 0;
options.beta = 1e-2;

options.Z = ones(D,6);

for d = 1:D
    options.bias(d) = yT{d}(1);
    options.scale(d) = std(yT{d});
end

%% Training IBPLFM variational approach
seeds = [1e5, 1e6, 1e4, 1e2, 1e1, 1e3, 8e6, 3e5, 6e4, 7e2];

Ni = 10;
LBres = zeros(Ni,1);
K = zeros(Ni,1);

parfor r = 1:Ni
    s = RandStream('mt19937ar', 'Seed', seeds(r));
    RandStream.setGlobalStream(s);
    [model, ll] = TrainSparseMGP(y, x, options);
    LBres(r,:) = ll(end);
    K(r) = sum(sum(round(model.etadq))>=1);
    ZT(r,:) = model.etadq(:)';
    ST(r,:) = model.kern.sensitivity(:)';
    savemodel(model,r,name);
end

K2 = zeros(1,10);
for r = 1:10
    Ztemp = reshape(ZT(r,:), D, options.nlf);
    Stemp = reshape(ST(r,:), D, options.nlf);
    tempZ = find(sum(abs(Ztemp) >= 3e-1) >= 1);
    tempS = find(sum(abs(Stemp) >= 1e-1) >= 1);
    temp = intersect(tempZ, tempS);
    K2(r) = length(temp);
end

save(strcat('temp/',name,'.mat'),'K','LBres','K','K2','ZT','ST','options');

%% Plot results from the selected solution
load(strcat('temp/',name,'.mat'));
[~, R] = max(LBres(:));
load(strcat('temp/',name,num2str(R),'.mat'));

etadq = zeros(size(model.etadq));
etadq(:,1:K(R)) = model.etadq(:,1:K(R));
model.etadq = round(etadq);

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
hinton(model.kern.sensitivity(:,1:K(R)));