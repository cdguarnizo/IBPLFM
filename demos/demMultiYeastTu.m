% DEMTOYIBPSIM Variational LFM with IBP prior over latent forces over the LFM on Yeast Spellman Data 
% DESC
% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel','../globalkern',genpath('../toolbox'),'../utils')

% Yeast data from TU
load ../datasets/Yeast/Tudata_G8.mat
names = {'YLR183C','YLR030W','TKL2','YOR359W','PFK27','RPL17B','RPS16B', ...
    'RPL13A'}; 
t = t./350;

test_ind{1} = t(1,:) >= t(1,9) & t(1,:) <= t(1,17);
test_ind{2} = t(5,:) >= t(5,16) & t(5,:) <= t(5,25);

outs = [1,5];
nout = length(outs);
D = size(f,1);
y = cell(D,1);
x = cell(D,1);
xT = cell(D,1);
yT = cell(D,1);
for d = 1:D,
    y{d} = f(d,:)';
    x{d} = t(d,:)';
    yT{d} = y{d};
    xT{d} = x{d};
    if any(d == outs),
        ind = find(d==outs);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.kernType = 'sim';
options.optimiser = 'scg2';

options.nlf = 8;
options.nout = D;
options.numActive = 18;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 0;
options.beta = 1e-2;

for d= 1:D,
    options.bias(d) = mean(yT{d});
    options.scale(d) = std(yT{d});
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