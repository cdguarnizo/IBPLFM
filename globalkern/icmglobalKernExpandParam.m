function kern = icmglobalKernExpandParam(kern, params)

%  ICMGLOBALKERNEXPANDPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010.
% MODIFICATIONS: C. Guarnizo, 2017.
% MGP

nParamsLat = kern.nlf;
kern.inverseWidth = reshape(params(1:nParamsLat), 1, kern.nlf);

startVal = nParamsLat+1;

kern.sensitivity = reshape(params(startVal:end), kern.nout, kern.nlf);