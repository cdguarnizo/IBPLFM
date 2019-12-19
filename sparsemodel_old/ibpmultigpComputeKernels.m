function model = ibpmultigpComputeKernels(model)

% IBPMULTIGPCOMPUTEKERNELS

% IBPMULTIGP

fhandle = str2func([model.kernType 'KernCompute']);
if isfield(model, 'gamma') && ~isempty(model.gamma)
    [model.Kff, model.Kfu, model.KuuGamma] = fhandle(model.kern, ...
        model.outX, model.latX, model.gamma);
else
    [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
        model.outX, model.latX);
end
model = ibpmultigpUpdateA2(model); %Update variables related to Kff Kfu and Kuu and beta