function [KernOutParams S] = ExtractGPmatParams(params, kernType, D, nlf)
if strcmp(kernType,'lfm'),
    np=5; %lengthscales mass spring dampes
    KernOutParams = [params(1:np*D+2:nlf*(np*D+1)+1) ...
        params(3:np:np*(D-1)+3) params(4:np:np*(D-1)+4) params(5:np:np*(D-1)+5)];
elseif strcmp(kernType,'sim'),
    np=3; %lengthscales variance decay
    KernOutParams = [params(1:np*D+2:nlf*(np*D+1)+1) ...
        params(3:np:np*(D-1)+3)];
else
    np=4; %lengthscales precU precG
    KernOutParams = [params(1:np*D+2:nlf*(np*D+1)+1) ...
        params(4:np:np*(D-1)+4)];
end

%Extract sensitivities
S = zeros(D, nlf);
start = 2 + np;
for q = 1:nlf,
    S(:,q) = params(start:np:start+(D-1)*np);
    start = start+(D-1)*np + (2 + np);
end