function g = ibpmultigpGradient(params, model)

% IBPMULTIGPGRADIENT 

% IBPMULTIGP

try
    model = ibpmultigpExpandParam(model, params);
    model = ibpmultigpComputeKernels(model); %Update Kff Kfu and Kuu
    g = -ibpmultigpLowerBoundGradients(model);
catch
    g = zeros(size(params));
end