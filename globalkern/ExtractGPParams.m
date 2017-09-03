function [KernOutParams S] = ExtractGPParams(params, kernType)
if strcmp(kernType,'lfm'),
    %lengthscales mass spring damper
    KernOutParams = params([4, 1:3]);
    S = params(5);
elseif strcmp(kernType,'sim'),
    % lengthscales decay
    KernOutParams = params([2 1]);
    %Sensitivities are not included, are known as variance
    S = params(3);
else
    %PrecU PrecG
    KernOutParams = params(1:2);
    %Sensitivity
    S = params(4);
end