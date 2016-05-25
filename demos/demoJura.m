% DEMTOJURA Variational LFM with IBP prior over latent force for Jura Data
% DESC
% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../toolbox/gpmat')

% load data Predict Cd from Ni and Zn
load ../datasets/Jura/juraDataCd.mat

D = size(y,1);

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'gg';
options.fixinducing = false;
options.IBPisInfinite = false;

options.nlf = 4;
options.numActive = 200;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 1;
options.beta = 1e-2;

options.InitInvWidth = 100;

[model, ll, mae, mse, msmse , mmsll] = TrainIBPLFM(y, X, yTest, XTest, options);