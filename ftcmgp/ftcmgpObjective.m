function f = ftcmgpObjective(params, model)

% IBPMULTIGPOBJECTIVE Wrapper function for MODELOPTIMISE objective.

% IBPMULTIGP
model = ftcmgpExpandParam(model, params);
model = ftcmgpComputeKernel(model); %Update Kff
f = -ftcmgpLowerBound(model);