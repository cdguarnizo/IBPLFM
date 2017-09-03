function kern = lfmglobalKernExpandParam(kern, params)

%LFMGLOBALKERNEXPANDPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010.
% MODIFICATIONS: C. Guarnizo, 2015.
% MGP

nParamsLat = kern.nlf;
kern.inverseWidth = reshape(params(1:nParamsLat), 1, kern.nlf);
%kern.inverseWidth(kern.inverseWidth > 150.) = 150;
%kern.inverseWidth(kern.inverseWidth < 1e-2) = 1e-2;

nParamsOut = kern.nout; %Mass_d Spring_d and Damper_d


startVal = nParamsLat+1;
if (isfield(kern, 'incMass') && kern.incMass),
    kern.mass = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
    startVal = startVal + nParamsOut;
end

kern.spring = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
%kern.spring(kern.spring > 20.) = 20;
%kern.spring(kern.spring < 1e-2) = 1e-2;

startVal = startVal + nParamsOut;
kern.damper = reshape(params(startVal:startVal-1+nParamsOut), 1, kern.nout);
%kern.damper(kern.damper > 20.) = 20;
%kern.damper(kern.damper < 1e-2) = 1e-2;
startVal = startVal + nParamsOut;

if ~(isfield(kern, 'isVarS') && kern.isVarS),
    kern.sensitivity = reshape(params(startVal:end), kern.nout, kern.nlf);
end