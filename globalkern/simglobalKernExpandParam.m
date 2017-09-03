function kern = simglobalKernExpandParam(kern, params)

% GGGLOBALKERNEXPANDPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010

% MULTIGP


nParamsLat = kern.nlf;
kern.inverseWidth = reshape(params(1:nParamsLat), 1, kern.nlf);
nParamsOut = kern.nout;
kern.decay = reshape(params(nParamsLat+1:nParamsLat+nParamsOut), 1, kern.nout);
if ~(isfield(kern, 'isVarS') && kern.isVarS),
    kern.sensitivity = reshape(params(nParamsLat+nParamsOut+1:end), kern.nout, kern.nlf);
end