function g = ftcmgpGradient(params, model)

% FTCMULTIGPGRADIENT 
% FTCMULTIGP

model = ftcmgpExpandParam(model, params);
model = ftcmgpComputeKernel(model);
g = -ftcmgpLowerBoundGradients(model);