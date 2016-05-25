function f = ibpmultigpObjective(params, model)

% IBPMULTIGPOBJECTIVE Wrapper function for MODELOPTIMISE objective.

% IBPMULTIGP
model = ibpmultigpExpandParam(model, params);
model = ibpmultigpComputeKernels(model); %Update Kff Kfu and Kuu
%model = ibpmultigpComputeELog(model);
%model = ibpmultigpMomentEuast(model);
f = -ibpmultigpLowerBound(model);