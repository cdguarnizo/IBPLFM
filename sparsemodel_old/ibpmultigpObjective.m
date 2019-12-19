function f = ibpmultigpObjective(params, model)

% IBPMULTIGPOBJECTIVE Wrapper function for MODELOPTIMISE objective.

% IBPMULTIGP
try
    model = ibpmultigpExpandParam(model, params);
    model = ibpmultigpComputeKernels(model); %Update Kff Kfu and Kuu
    f = -ibpmultigpLowerBound(model);
catch
    f = inf;
end