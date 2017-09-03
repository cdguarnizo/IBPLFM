% DEMTOYIBPLFM Variational LFM with IBP prior over latent forces

% MULTIGP

clc
clear
close all
format short e

addpath('../sparsemodel22',genpath('../toolbox'));

% load data
fd = amc_to_matrix('../datasets/CMUmocap/02_03.amc');

% 8 signals test
% humerus radius femur foot
IndSel = [27,30,49,53,39,42,56,60];
% humerus radius femur tibia
% IndSel = [27,30,49,52,39,42,56,59];

outs = [1,4,6];

% 6 channels
% humerus radius femur tibia
% IndSel = [27,30,49,39,42,56];
% outs = [1,5];

fd = fd(:,IndSel);
[N, D] = size(fd);
t = (1:N)'/120; %Time stamp in seconds
%Downsample
IndDown = 1:4:N;

%test_ind{1} = t(IndDown) >= .5 & t(IndDown) <= .9;
%test_ind{2} = t(IndDown) >= .08 & t(IndDown) <= .5;

test_ind{1} = t(IndDown) >= .9 & t(IndDown) <= 1.2;
test_ind{2} = t(IndDown) >= .5 & t(IndDown) <= .9;
test_ind{3} = t(IndDown) >= .08 & t(IndDown) <= .5;


y = cell(D,1);
x = cell(D,1);
xT = cell(D,1);
yT = cell(D,1);
for d = 1:D,
    y{d} = fd(IndDown, d);
    x{d} = t(IndDown);
    yT{d} = y{d};
    xT{d} = x{d};
    if any(d == outs),
        ind = find(outs==d);
        y{d}(test_ind{ind}) = [];
        x{d}(test_ind{ind}) = [];
    end
end

clear fd t

% Set the Options 
options = ibpmultigpOptions('dtcvar');
options.sparsePriorType = 'ibp';
options.kernType = 'lfm';
options.optimiser = 'scg';

options.isVarS = false; %If ARD or SpikeSlab this should be true
options.gammaPrior = false;
options.InitSearchS = false;

options.fixinducing = true;
options.Trainkern = true;
options.InitKern = false;
options.debug = false;

options.sorteta = true; %If ARD then this should be false
options.isVarU = true;
options.OptMarU = true;
options.IBPisInfinite = true;
options.Opteta = false;
options.force_posUpdate = false;

options.nlf = 10;
options.numActive = 25;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 0;
options.beta = 1e-2;

options.UseMeanConstants = false;
for d = 1:D,
    %options.bias(d) = yT{d}(1);
    options.bias(d) = mean(yT{d});
    options.scale(d) = std(yT{d});
end

seeds = [1e5, 1e6, 1e4, 1e2, 1e1, 1e3, 8e6, 3e5, 6e4, 7e2];

%isVarS InitSearchS InitKern OptMarU
testflag = [1,1,1,1,0,0,0,0; 0,1,1,0,0,0,0,0; 0,0,0,1,1,0,0,1; 0,0,0,0,0,0,1,1];
testbeta = [1e-3, 1e-3, 1e3, 1e-2, 1e-2, 1e-2, 1e-2 1e-2];
%for c = 8,
    c = 8;
    results = zeros(10,3);
    eta = zeros(10, options.nlf*D);
    S = eta;
    K = zeros(10,1);
    fail = cell(10,1);
    llt = zeros(10,options.NI);
    mu = zeros(10,D);
    %Prediction measures
    mae = zeros(10,length(outs));
    mse = zeros(10,length(outs));
    smse = zeros(10,length(outs));
    msll = zeros(10,length(outs));

    options.isVarS = testflag(1, c);
    options.InitSearchS = testflag(2, c);
    options.InitKern = testflag(3, c);
    options.OptMarU = testflag(4, c);
    options.beta = testbeta(c);
    
    modstr = '';
    if options.isVarS,
        modstr = strcat(modstr,'_varS');
    end
    
    if options.InitKern,
        modstr = strcat(modstr,'_initK');
    end
    
    if options.OptMarU,
       modstr = strcat(modstr,'_optMarU');
    end 
    
    if options.force_posUpdate,
        modstr = strcat(modstr,'_PosU');
    end
    
    if  options.IBPisInfinite,
        modstr = strcat(modstr,'_inf');
    else
        modstr = strcat(modstr,'_fin');
    end
    
    if options.sorteta,
        modstr = strcat(modstr,'_soreta');
    end

    name = strcat('/home/guarni/MEGA/ResultsIBPLFM/IBP',options.kernType,'_MOCAP_02_03_Ch_',...
        num2str(D),'_C',num2str(c),modstr,'_new.mat');
    
    parfor con = 1:10,
        s = RandStream('mt19937ar', 'Seed', seeds(con));
        RandStream.setGlobalStream(s);
        [model, ll] = TrainSparseMGP2(y, x, options);

        [ymean, yvar] = ibpmultigpPosterior(model, xT);
        
        mae1 = zeros(1,length(outs));
        mse1 = zeros(1,length(outs));
        smse1 = zeros(1,length(outs));
        msll1 = zeros(1,length(outs));
        for k = 1:length(outs),
            ym = ymean{outs(k)}(test_ind{k});
            yv = yvar{outs(k)}(test_ind{k});
            
            [mae1(k), mse1(k), smse1(k), msll1(k), ~] = multigpErrorMeasures({y{outs(k)}},...
                {yT{outs(k)}(test_ind{k})}, {ym}, {yv}, 1);
        end
        
        llt(con,:) = ll;
        msmse = mean(smse1);
        mmsll = mean(msll1);
        results(con,:) = [ll(end), msmse, mmsll];
        
        mae(con, :) = mae1;
        mse(con, :) = mse1;
        smse(con,:) = smse1;
        msll(con,:) = msll1;
        
        if strcmp(options.sparsePriorType,'ard'),
            eta(con, :) = ones(1, options.nlf*D);
            K(con) = options.nlf;
        else
            eta(con, :) = model.etadq(:)';
            K(con) = sum(sum(round(model.etadq))>=1);
        end
        if options.isVarS,
            S(con,:) = model.muSdq(:)';
        else
            S(con,:) = model.kern.sensitivity(:)';
        end
        if options.UseMeanConstants,
            mu(con,:) = model.mu';
        end
        savemodel(model,ymean,yvar,con,options);
        model = [];
    end
    save(name,'seeds','results', 'eta','S','K','fail','llt','mu','mae','mse','smse','msll');
%end