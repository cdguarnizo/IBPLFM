function [f,g] = ibpmultigpObjectiveGradient(params, model)

model = ibpmultigpExpandParam(model, params);
model = ibpmultigpComputeKernels(model);
%model = ibpmultigpComputeELog(model);
%model = ibpmultigpUpdateA2(model);
f = -ibpmultigpLowerBoundOptimise(model);
g = -ibpmultigpLowerBoundGradients(model);