% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')

% load data
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
for d = 1:D,
    y{d} = fd(:,d);
    x{d} = xTemp;
    yT{d} = fd(:,d);
    xT{d} = xTemp;
    if any(d == outs),
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

for d = 1:D,
    options.bias(d) = 0;
    options.scale(d) = 1;
end

%% Training IBPLFM variational approach
seeds = [1e5, 1e6, 1e4, 1e2, 1e1, 1e3, 8e6, 3e5, 6e4, 7e2];

Ni = 10;
LBres = zeros(Ni,1);
K = zeros(Ni,1);

parfor con = 1:Ni,
    s = RandStream('mt19937ar', 'Seed', seeds(con));
    RandStream.setGlobalStream(s);
    [model, ll] = TrainSparseMGP(y, x, options);
    LBres(con,:) = ll(end);
    K(con) = sum(sum(round(model.etadq))>=1);
    savemodel(model,con);
end
save('temp/res.mat','K','LBres');
%% Plot results from the selected solution
load('temp/res.mat');
[~, R] = max(LBres(:));
load(strcat('temp/m',num2str(R),'.mat'));

etadq = zeros(size(model.etadq));
etadq(:,1:K(R)) = model.etadq(:,1:K(R));
model.etadq = etadq;

[ymean, yvar] = ibpmultigpPosterior(model, xT);

nmse = zeros(1,nout);
nlpd = zeros(1,nout);
fprintf('Best solution performance per output\n');
for k = 1:nout,
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

