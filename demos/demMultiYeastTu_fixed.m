% DEMTOYIBPSIM Variational LFM with IBP prior over latent forces over the LFM on Yeast Spellman Data 
% DESC
% MULTIGP

clc
clear
close all
format short e

addpath('../fixedmodel','../toolbox/gpmat')

% Yeast data from TU
load ../datasets/Yeast/Tudata_G8.mat

t = t./350;

test_ind{1} = t(1,:) >= t(1,9) & t(1,:) <= t(1,17);
test_ind{2} = t(5,:) >= t(5,16) & t(5,:) <= t(5,25);
outs = [1,5];

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
options.sparsePriorType = 'ibp';
options.kernType = 'sim';
options.optimiser = 'scg';

options.isVarS = false; %If ARD or SpikeSlab this should be true
options.gammaPrior = false;
options.InitSearchS = false;

options.fixinducing = true;
options.Trainkern = true;
options.InitSearchS = false;
options.InitKern = false;
options.debug = false;

options.sorteta = true; %If ARD then this should be false
options.isVarU = true;
options.OptMarU = true;
options.IBPisInfinite = true;
options.force_posUpdate = false;
options.Opteta = false;

options.nlf = 2;
options.nout = D;
options.numActive = 18;
options.alpha = 1;
options.NI = 200;
options.NIO = 20;
options.DispOpt = 2;
options.beta = 1e-2;
options.Z = [1,1,1,0,0,1,1,1;0,0,0,1,1,1,1,1]';


%options.PosS = true;
options.UseMeanConstants = false;
for d= 1:D,
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
    llt = zeros(10,1);
    mu = zeros(10,D);
    %Prediction measures
    mae = zeros(10,2);
    mse = zeros(10,2);
    smse = zeros(10,2);
    msll = zeros(10,2);

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

    name = strcat('/home/guarni/MEGA/ResultsIBPLFM/IBP',options.kernType,'_TuG8_M',...
        num2str(options.numActive),'_C',num2str(c),modstr,'.mat');
    
    for con = 1:10,
        s = RandStream('mt19937ar', 'Seed', seeds(con));
        [model, ll] = TrainFixedmlfm(y, x, options);

        [ymean, yvar] = ibpmultigpPosterior(model, xT);
        
        mae1 = zeros(1,2);
        mse1 = zeros(1,2);
        smse1 = zeros(1,2);
        msll1 = zeros(1,2);
        for k = 1:2,
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
        
        savemodel(model,ymean,yvar,con,options);
        model = [];
    end
    %save(name,'seeds','results', 'eta','S','K','fail','llt','mu','mae','mse','smse','msll');
%end