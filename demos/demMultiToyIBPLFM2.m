% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')

% load data
name = 'IBPLFM';
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

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'lfm';
options.optimiser = 'scg2';

options.nlf = 4;
options.numActive = 25;
options.alpha = 1;
options.NI = 300;
options.NIO = 20;
options.DispOpt = 0;
options.beta = 1e-2;

for d = 1:D
    options.bias(d) = 0;
    options.scale(d) = 1;
end

%% Training IBPLFM variational approach
seeds = [1e5, 1e6, 1e4, 1e2, 1e1, 1e3, 8e6, 3e5, 6e4, 7e2];

Ni = 10;
LBres = zeros(Ni,300);
K = zeros(Ni,1);
ZT = zeros(10,D*options.nlf);
ST = zeros(10,D*options.nlf);
parfor r = 1:Ni
    s = RandStream('mt19937ar', 'Seed', seeds(r));
    RandStream.setGlobalStream(s);
    [model, ll] = TrainSparseMGP(y, x, options);
    LBres(r,:) = ll;
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

save('Toy','K','K2','LBres','ZT','ST','options','name');

%%
% Plot results from the selected solution
load('Toy.mat');
[~, R] = max(LBres(:,end));
load(strcat('temp/',name,num2str(R),'.mat'));

etadq = zeros(size(model.etadq));
etadq(:,1:K(R)) = model.etadq(:,1:K(R));
etadq = etadq.*(round(etadq));
model.etadq = etadq;

S = zeros(size(model.kern.sensitivity));
S(:,1:K(R)) = model.kern.sensitivity(:,1:K(R));
S = S.*(abs(S) >= 1e-1);
model.kern.sensitivity = S;

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

[umean, uvar] = ibpmultigpPosteriorLatent(model, xT{1});
figure
%plot(xT{1},uq(:,1)/max(uq(:,1)),xT{1},umean{3}/max(umean{3}))
plotGP(model.kern.sensitivity(1,2)*umean{2}, uvar{2}*model.kern.sensitivity(1,2)^2, xT{1})
figure
%plot(xT{1},uq(:,2)/max(uq(:,2)),xT{1},umean{1}/max(umean{1}))
plotGP(model.kern.sensitivity(1,1)*umean{1}, model.kern.sensitivity(1,1)^2*uvar{1}, xT{1})

%Plot hinton diagram
hinton(model.kern.sensitivity(:,[2,1]).*model.etadq(:,[2,1]));

%%
for k=1:10
    hinton(reshape(ZT(k,:).*ST(k,:),3,4))
end