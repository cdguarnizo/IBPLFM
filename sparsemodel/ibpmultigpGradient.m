function g = ibpmultigpGradient(params, model)

% IBPMULTIGPGRADIENT 

% IBPMULTIGP

%model = modelExpandParam(model, params);
%g =  ibpmultigpLowerBoundGradients(model);

model = ibpmultigpExpandParam(model, params);
%model = modelExpandParam(model, params);
model = ibpmultigpComputeKernels(model);
%model = ibpmultigpComputeELog(model);
%model = ibpmultigpUpdateA2(model);
%model = ibpmultigpMomentEuast(model);
g = -ibpmultigpLowerBoundGradients(model);