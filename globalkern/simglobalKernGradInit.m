function kern = simglobalKernGradInit(kern)

% GGGLOBALKERNGRADINIT
%
% COPYRIGHT : Mauricio A. Alvarez, 2010
% MODIFICATIONS: Cristian Guarnizo, 2014
% MULTIGP

kern.grad.inverseWidth = zeros(size(kern.inverseWidth));
kern.grad.decay = zeros(size(kern.decay));