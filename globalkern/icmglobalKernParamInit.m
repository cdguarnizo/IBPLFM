function kern = icmglobalKernParamInit(kern)

% LFMGLOBALKERNPARAMINIT
%
% COPYRIGHT : Mauricio A. Alvarez, 2010, 2013
% MODIFICATIONS: Cristian Guarnizo, 2014
% MULTIGP

if isfield(kern, 'options') && isfield(kern.options, 'nout')
    kern.nout = kern.options.nout;
else
    error('Number of outputs is required for this kernel')
end

if isfield(kern, 'options') && isfield(kern.options, 'nlf')
    kern.nlf = kern.options.nlf;    
else
    error('Number of latent forces is required for this kernel')
end

if isfield(kern, 'options') && isfield(kern.options, 'approx')
    kern.approx = kern.options.approx;    
else
    error('Approximation method is required for this kernel')
end

if ~isfield(kern, 'inputDimension')
    warning('lfmglobalKernParamInit:noInputDimension', 'Input dimension has not been provided. Assuming is one.')
    kern.inputDimension = 1;
end

kern.isArd = false;

if isfield(kern, 'options') && isfield(kern.options, 'tieOutputParams'),
    kern.tieOutputParams = kern.options.tieOutputParams;
    if kern.options.tieOutputParams
        kern.inverseWidth = 2./(2:2-.5/kern.nlf:.1).^2;
        kern.nParams = kern.nlf;
    else
        error('Not implemented yet.');
    end
else
    kern.tieOutputParams = true;
    kern.inverseWidth = 2./(.5 + rand(1, kern.nlf)).^2;
    kern.nParams = kern.nlf;
end
kern.lfParamsTemplate = 1;
kern.outParamsTemplate = 1;

kern.sensitivity = ones(kern.nout, kern.nlf);
if isfield(kern, 'options') && isfield(kern.options, 'isVarS') ...
        && kern.options.isVarS,
    kern.isVarS = kern.options.isVarS;
else
    kern.isVarS = false;
end

kern.transforms.index = 1:kern.nParams;
if ~kern.isVarS,
    kern.transforms.index = 1:kern.nParams;
    kern.nParams = kern.nParams + kern.nout*kern.nlf;
end
kern.transforms.type = optimiDefaultConstraint('positive');
kern.isStationary = true;