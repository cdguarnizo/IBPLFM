function Kff = ggglobalKffCompute(kern, outX, d, q)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP

kernOut = kern.template.output;

% Expand the parameter decay
kernOut.precisionG = kern.precisionG(d);
kernOut.sensitivity = kern.sensitivity(d,q);
% Expand the parameter inverseWidth
kernOut.precisionU = kern.precisionU(q);
% Compute Kff
Kff = real(kern.funcNames.computeOut(kernOut, outX));

if any(isnan(Kff)) | any(isinf(Kff)),
    error('Nan or Inf in Kff')
end