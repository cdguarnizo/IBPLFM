function kern = lfmglobalKernGradInit(kern)

% GGGLOBALKERNGRADINIT
%
% COPYRIGHT : Mauricio A. Alvarez, 2010
% MODIFICATIONS: Cristian Guarnizo, 2014
% MULTIGP

kern.grad.inverseWidth = zeros(size(kern.inverseWidth));
kern.grad.mass = zeros(size(kern.mass));
kern.grad.spring = zeros(size(kern.spring));
kern.grad.damper = zeros(size(kern.damper));