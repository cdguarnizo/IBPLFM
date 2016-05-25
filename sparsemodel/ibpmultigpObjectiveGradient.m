function [f,g] = ibpmultigpObjectiveGradient(params, model)

model = ibpmultigpExpandParam(model, params);
model = ibpmultigpComputeKernels(model);
%model = ibpmultigpComputeELog(model);
%model = ibpmultigpUpdateA2(model);
f = -ibpmultigpLowerBound(model);
g = -ibpmultigpLowerBoundGradients(model);