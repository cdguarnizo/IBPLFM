function kern = lfmglobalKernExpandParam(kern, params)

%LFMGLOBALKERNEXPANDPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010.
% MODIFICATIONS: C. Guarnizo, 2015.
% MGP

nParamsLat = kern.nlf;
kern.inverseWidth = reshape(params(1:nParamsLat), 1, kern.nlf);

nParamsOut = kern.nout; %Mass_d Spring_d and Damper_d


startVal = nParamsLat+1;
if (isfield(kern, 'incMass') && kern.incMass),
    kern.mass = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
    startVal = startVal + nParamsOut;
end

kern.spring = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
startVal = startVal + nParamsOut;
kern.damper = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
startVal = startVal + nParamsOut;

if ~(isfield(kern, 'isVarS') && kern.isVarS),
    kern.sensitivity = reshape(params(startVal:end), kern.nout, kern.nlf);
end