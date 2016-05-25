% DEMOMOCAP_02 Variational LFM with IBP prior over latent forces for MOCAP
% data.

%% Initialization
clc
clear
close all
format short e

addpath('../sparsemodel',genpath('../toolbox'));

%% load data
fd = amc_to_matrix('../datasets/CMUmocap/02_01.amc');
IndSel = [5 6 8 9 11 14:20 22 23 27:31 39:43 49 51:53 55 56 58 59];
IndSel = IndSel(1:15);
fd = fd(:,IndSel);

[N, D] = size(fd);
x = (1:N)'/120; %Time stamp in seconds

interv = round(N/100);
TrainInd = 1:interv:N;
TestInd = 1:N;
TestInd(TrainInd) = [];

yTemp2 = cell(D,1);
xTemp2 = cell(D,1);
xTemptest = cell(D,1);
yTemptest = cell(D,1);

for d = 1:D,
    yTemp2{d} = fd(TrainInd, d);
    xTemp2{d} = x(TrainInd);
    yTemptest{d} = fd(TestInd, d);
    xTemptest{d} = x(TestInd);
end

clear fd x

%% Set IBPLFM Options 
options = ibpmultigpOptions('dtcvar');
options.sparsePriorType = 'ibp';
options.kernType = 'lfm';
options.fixinducing = true;
options.IBPisInfinite = true;

%Maximum number of latent forces
options.nlf = 4;
%Number of inducing poitns
options.numActive = 25;
%Set IBP parameter value
options.alpha = 2;
%Maximum number of iterations for EM algorithm
options.NI = 200;
%Number of iteration for hyperparameters optimization
options.NIO = 20;
%Show hyperparameter optimization performance
options.DispOpt = 1;
%Initial value for precision of noises
options.beta = 1e-2;

%% Train IBPLFM
[model, ll, mae, mse, msmse , mmsll] = TrainIBPLFM(yTemp2, xTemp2, yTemptest, xTemptest, options);

%% Testing data prediction

%% Latent forces prediction