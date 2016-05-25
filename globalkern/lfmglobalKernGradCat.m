function g = lfmglobalKernGradCat(kern)

% GGGLOBALKERNGRADCAT
%
% COPYRIGHT : Mauricio A. Alvarez, 2010
% MODIFICATIONS: Cristian Guarnizo, 2014.
% MULTIGP

g = [kern.grad.inverseWidth(:)' kern.grad.spring(:)' kern.grad.damper(:)'];