function m = spmultimodelComputeM(model)

% SPMULTIMODELCOMPUTEM 
%
% COPYRIGHT : Mauricio A. Alvarez, 2010

% MULTIGP


if isfield(model, 'meanFunction') && ~isempty(model.meanFunction)
    mu = meanCompute(model.meanFunction, mat2cell(ones(model.nout,1), ones(model.nout,1), 1), 0);
else
    mu = zeros(model.nout,1);
end
m = cell(model.nout,1);
for j=1:model.nout,
    m{j} = model.y{j} - mu(j);
    if model.bias(j)~=0
        m{j} = m{j} - model.bias(j);
    end
    if model.scale(j)~=1
        m{j} = m{j}/model.scale(j);
    end
end