% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')

% load data
name = 'weather';
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

options.nlf = 10;
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

parfor r = 1:Ni
    s = RandStream('mt19937ar', 'Seed', seeds(r));
    RandStream.setGlobalStream(s);
    [model, ll] = TrainSparseMGP(y, x, options);
    LBres(r,:) = ll(end);
    ZT(r,:) = model.etadq(:)';
    ST(r,:) = model.kern.sensitivity(:)';
    K(r) = sum(sum(round(model.etadq))>=1);
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

save(strcat('temp/',name),'K','K2','LBres','ZT','ST','options','name');

%% Plot results from the selected solution
load(strcat('temp/',name,'.mat'));
[~, R] = max(LBres(:));
load(strcat('temp/',name,num2str(R),'.mat'));

etadq = round(model.etadq)
model.etadq = etadq;

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
hinton(model.etadq(:,1:K(R)).*model.kern.sensitivity(:,1:K(R)));