addpath('lfm2');

%Bal
fd = amc_to_matrix('02_01.amc');
IndSel = [5 6 8 9 11 14:20 22 23 27:31 39:43 49 51:53 55 56 58 59];

fd = fd(:,IndSel(1:15));

%Wal
%fd = amc_to_matrix('35_01.amc');
%IndSel = [1 6 8 9 11 12 14 15 27 28 29 30 39 40 41 42 49 51 56 58];



% %Normalize data
for d = 1:size(fd,2),
    fd(:,d) = (fd(:,d)-mean(fd(:,d)))/std(fd(:,d));
    fd(:,d) = fd(:,d) - fd(1,d);
end

[N D]=size(fd);
xTemp = (1:N)/120; %Time stamp in seconds15
xTemp = xTemp(:);

%Randomly select the frames
interv = round(N/45);
%Nt = N + (interv - mod(N,interv));
%TrainInd = ceil(interv*rand(Nt/interv,1));
%TrainInd(1,1) = 1;
%TrainInd(end,1) = mod(N,interv);
%SampInd = reshape(1:Nt,interv,ceil(N/interv))';
%SelecInd = SampInd(sub2ind(size(SampInd),(1:Nt/interv)', TrainInd));
SelecInd = 1:interv:N;
TestInd = 1:N;
TestInd(SelecInd) = [];
options.nout = D;
yTemp2 = cell(1,options.nout);
xTemp2 = cell(1,options.nout);
xTemptest = cell(1,options.nout);
yTemptest = cell(1,options.nout);
for d = 1:options.nout
    yTemp2{d} = fd(SelecInd, d);
    %yTemp2{d} = (yTemp2{d}-mean(yTemp2{d}))/std(yTemp2{d});
    %yTemp2{d} = yTemp2{d} - fd(1,d);
    xTemp2{d} = xTemp(SelecInd);
    yTemptest{d} = fd(TestInd, d); %- fd(1,d);
    xTemptest{d} = xTemp(TestInd);
end
X = xTemp2;
y = yTemp2;

%load resIBPLFM_Bal_D15Q9.mat
load res3BalIBPGG_D15Q9.mat
model = modelT{1,1};

%load res2WalIBPGG_D5Q4.mat


model.sparsePriorType = 'ibp';
model.actyhat =  true;
[ymean yvar]=ibpmultigpPosterior(model, xTemptest);

[mae, mse, smse, msll] = multigpErrorMeasures(yTemp2, yTemptest, ymean, ...
    yvar, model.nout);
msmse = mean(smse);
mmsll = mean(msll);