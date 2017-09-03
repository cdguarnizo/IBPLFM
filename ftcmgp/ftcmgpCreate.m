function model = ftcmgpCreate( X, y, options )

% FTCMGPCREATE
% Inputs:
% X: Input data (time stamps and output indexation)
% y: Output data
% Options: Definition of flag variables and type of inference.
% FTCMGP

model.type = 'ftcmgp';

model.nout = max(X.ind);
model.kernType = options.kernType;
model.ndim = size(X.val,2);
model.nlf = options.nlf;
model.approx = options.approx;
model.optimiser = options.optimiser;
model.isVarS = options.isVarS;

if isfield(options, 'scale') && ~isempty(options.scale)
    model.scale = options.scale;
else
    model.scale = ones(1, model.nout);
end
if isfield(options, 'bias') && ~isempty(options.bias)
    model.bias = options.bias;
else
    model.bias = zeros(1, model.nout);
end

model.includeNoise = options.includeNoise;

model.outX = X;
model.y = y;
kern.type = [options.kernType 'global'];
kern.inputDimension = model.ndim;
kern.options = options.kern;
kern.options.nlf = model.nlf;
kern.options.nout = model.nout;
kern.options.approx = model.approx;
kern.options.isVarS = options.isVarS;

kern = kernParamInit(kern);
kern.template.output = kernCreate(X.val(X.ind==1,:), options.kernType);
kern.funcNames.computeOut = str2func([options.kernType 'KernCompute']);
kern.funcNames.computeCross = str2func([options.kernType 'X' options.kernType 'KernCompute']);
kern.funcNames.gradientCross = str2func([options.kernType 'X' options.kernType 'KernGradient']);
kern.funcNames.gradientOut = str2func([options.kernType 'KernGradient']);
kern.funcNames.extractOut = str2func([options.kernType 'KernExtractParam']);
model.kern = kern;
model.kernType = kern.type;
model.kern.paramGroups = speye(model.kern.nParams);
numParams = model.kern.nParams;

% Count number of parameters
model.nParams = numParams;

% Set up a mean function if one is given.
if isfield(options, 'meanFunction') && ~isempty(options.meanFunction)
    if isstruct(options.meanFunction)
        model.meanFunction = options.meanFunction;
    else
        if ~isempty(options.meanFunction)
            model.meanFunction = meanCreate(model.ndim, model.nout, X, y, options.meanFunctionOptions);
        end
    end
    model.nParams = model.nParams + model.meanFunction.nParams;
end

% Create noise models
switch model.approx,
     case 'ftc'
        if isfield(options, 'beta') && ~isempty(options.beta)
            if size(options.beta,2) == model.nout
                model.beta = options.beta;
            else
                model.beta = options.beta*ones(1,model.nout);
            end
            model.betaTransform =  optimiDefaultConstraint('positive');
            model.nParams = model.nParams + model.nout;
        end
    case {'fitc','pitc', 'dtcvar'}
        % In this structure is easier to put noise in the latent functions
        % at the top level
        if isfield(options, 'gamma') && ~isempty(options.gamma)
            if size(options.gamma,2) == model.nlf
                model.gamma = options.gamma;
            else
                model.gamma = options.gamma*ones(1,model.nlf);
            end
            model.gammaTransform =  optimiDefaultConstraint('positive');
            model.nParams = model.nParams + model.nlf;
        end
        if isfield(options, 'beta') && ~isempty(options.beta)
            if size(options.beta,2) == model.nout
                model.beta = options.beta;
            else
                model.beta = options.beta*ones(1,model.nout);
            end
            model.betaTransform =  optimiDefaultConstraint('positive');
            model.nParams = model.nParams + model.nout;
        end
end
params = modelExtractParam(model);
model = modelExpandParam(model, params);