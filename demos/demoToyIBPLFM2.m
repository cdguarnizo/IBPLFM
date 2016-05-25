% DEMTOYIBPLFM2 Variational LFM with IBP prior over latent forces
% IBPLFM

%% Initialization
clc
clear
close all
format short e

addpath('../sparsemodel','../toolbox/gpmat')

%% Load data
load ../datasets/Toys/datasetD3Q2_IBP_LFM2.mat

nout = size(fd,2);
y = cell(nout,1);
x = cell(nout,1);
for d = 1:nout,
    y{d} = fd(:,d);
    x{d} = xTemp;
end
clear fd xTemp

%% Set IBPLFM Options 
options.kernType = 'lfm';
options.optimiser = 'scg';
options.fixinducing = true;
options.IBPisInfinite = true;

options.nlf = 4;
options.numActive = 25;
options.alpha = 2;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 1;
options.beta = 1e-2;

%% Train IBPLFM
[model, ll, mae, mse, msmse , mmsll] = TrainIBPLFM(y, x, y, x, options);

%% Plot Output Estimation
close all
[ymean yvar]=ibpmultigpPosterior(model, x);

plotpred('Output', ymean, x, yvar, y, x)

%% Plot Latent Forces estimation
[up qpv]=ibpmultigpPosteriorLatent(model,x{1});

plotpred('Latent', up, x{1}, qpv)

%% Plot Hinton Diagram
hinton(Zdq.*Sdq)

hinton(model.etadq.*model.kern.sensitivity)