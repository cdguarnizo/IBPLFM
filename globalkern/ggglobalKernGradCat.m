function g = ggglobalKernGradCat(kern)

% GGGLOBALKERNGRADCAT
%
% COPYRIGHT : Mauricio A. Alvarez, 2010

% MULTIGP

if isfield(kern, 'isVarS') && kern.isVarS,
    g = [kern.grad.precisionU(:)' kern.grad.precisionG(:)'];
else
    g = [kern.grad.precisionU(:)' kern.grad.precisionG(:)' kern.grad.sensitivity(:)'];
end