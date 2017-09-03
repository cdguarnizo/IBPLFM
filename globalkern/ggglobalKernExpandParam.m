function kern = ggglobalKernExpandParam(kern, params)

% GGGLOBALKERNEXPANDPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010

% MULTIGP

if kern.isArd,
    nParamsLat = kern.inputDimension*kern.nlf;     
    kern.precisionU = reshape(params(1:nParamsLat), kern.inputDimension, kern.nlf);    
    if kern.tieOutputParams,
        nParamsOut = kern.inputDimension*kern.out;
        kern.precisionG = reshape(params(nParamsLat+1:nParamsLat+nParamsOut), kern.inputDimension, kern.nout);    
    else
        nParamsOut = kern.inputDimension*kern.out*kern.nlf;
        kern.precisionG = reshape(params(nParamsLat+1:nParamsLat+nParamsOut), kern.inputDimension, kern.nout, kern.nlf);    
    end
else
    nParamsLat = kern.nlf;
    kern.precisionU = reshape(params(1:nParamsLat), 1, kern.nlf);
    %kern.precisionU(kern.precisionU<1e-2) = 1e-2;
    %kern.precisionU(kern.precisionU>200) = 200;
    
    nParamsOut = kern.nout;
    kern.precisionG = reshape(params(nParamsLat+1:nParamsLat+nParamsOut), 1, kern.nout);
    %kern.precisionG(kern.precisionG < 1e-2) = 1e-2;
    %kern.precisionG(kern.precisionG > 200) = 200;
end

if ~(isfield(kern.options, 'isVarS') && kern.options.isVarS),
    kern.sensitivity = reshape(params(nParamsLat+nParamsOut+1:end), kern.nout, kern.nlf);
end