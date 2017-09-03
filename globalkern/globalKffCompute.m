function Kff = globalKffCompute(kern, outX, d, q)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP

kernOut = globalSetKernOut(kern,d,q);
% Compute Kff
Kff = real(kern.funcNames.computeOut(kernOut, outX));

if any(isnan(Kff)) | any(isinf(Kff)),
    error('Nan or Inf in Kff')
end