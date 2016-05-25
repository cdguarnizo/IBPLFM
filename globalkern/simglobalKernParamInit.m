function kern = simglobalKernParamInit(kern)

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
kern.isNegativeS =  true;

if isfield(kern, 'options') && isfield(kern.options, 'tieOutputParams'),
    kern.tieOutputParams = kern.options.tieOutputParams;
    if kern.options.tieOutputParams
        kern.inverseWidth = 1./(1 + .5*randn(1, kern.nlf)).^5;
        kern.decay = ones(1, kern.nout);
        kern.nParams = kern.nlf + kern.nout;
    else %Not implemented yet
        error('Not implemented yet.');
    end        
else
    kern.tieOutputParams = true;
    kern.inverseWidth = 1./(1 + .5*randn(1, kern.nlf)).^5;
    kern.decay = ones(1, kern.nout);
    kern.nParams = kern.nlf + kern.nout;
    kern.ParamsperOut = 1;
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
    %kern.diffParams = 2;
    kern.transforms.index = 1:kern.nParams;
    kern.nParams = kern.nParams + kern.nout*kern.nlf;
end
kern.transforms.type = optimiDefaultConstraint('positive');
kern.isStationary = false;