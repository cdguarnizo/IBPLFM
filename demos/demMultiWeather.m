% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')
% load data
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

options.nlf = 6;
options.numActive = 200;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 1;
options.beta = 1e-2;

%% Training IBPLFM variational approach
seeds = [1e5, 1e6, 1e4, 1e2, 1e1, 1e3, 8e6, 3e5, 6e4, 7e2];
%Number of models to be trained
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
etadq = model.etadq(:,1:R);
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