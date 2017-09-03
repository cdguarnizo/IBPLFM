function [f,g] = ibpmultigpObjectiveGradient(params, model)

try
    model = ibpmultigpExpandParam(model, params);
    model = ibpmultigpComputeKernels(model);
    f = -ibpmultigpLowerBound(model);
    g = -ibpmultigpLowerBoundGradients(model);
catch
    f = inf;
    g = zeros(size(params));
end