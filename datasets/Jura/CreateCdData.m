load train
load valdata
primaryLoc = 1; % Column location of 
secondaryLoc = [5 7]; % Column location of Ni and Zn
predValues = y(:,secondaryLoc);
valValues = ytest(:,secondaryLoc);
Xtrain = X;
ytrain = y;
X = cell(1, 1+length(secondaryLoc));
y = cell(1, 1+length(secondaryLoc));
XTest =  cell(1, 1+length(secondaryLoc));
yTest =  cell(1, 1+length(secondaryLoc));
X{1} = Xtrain;
y{1} = ytrain(:,primaryLoc);
for i=1:length(secondaryLoc);
    X{1+i} = Xtrain;
    y{1+i} = ytrain(:, secondaryLoc(i));
end
% Append validation values to training data
for i=1:length(secondaryLoc);
    X{1+i} = [X{1+i}; Xtest];
    y{1+i} = [y{1+i}; ytest(:, secondaryLoc(i))];
end
% Form the testting sets
XTest{1} = Xtest;
yTest{1} = ytest(:, primaryLoc);
for i=1:length(secondaryLoc);
    XTest{1+i} = Xtest;
    yTest{1+i} = ytest(:, secondaryLoc(i));
end
save juraDataCd X y XTest yTest