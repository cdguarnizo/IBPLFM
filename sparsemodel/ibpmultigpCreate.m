function model = ibpmultigpCreate(X, y, options )

% IBPMULTIGPCREATE
% Based on multigp code from GPmat. This function allows the creation of
% Sparse models, where sparcity is induced over the sensitivities of
% convolved Gaussian Processes.
% Inputs:
% X: Input data (cell type)
% y: Output data (cell typ)
% Options: Definition of flag variables and type of inference.
% IBPMULTIGP

model.type = 'ibpmultigp';

if isfield(options, 'varS') && ~isempty(options.varS)
    if ~strcmp(options.approx, 'dtcvar')
        error('Learning the sensitivities is only possible for dtcvar approximation.')
    end
    model.isVarS = options.isVarS;
else
    model.isVarS = false;
end

model.debug = options.debug;
model.nout = max(size(y));
model.kernType = options.kernType;
model.ndim = size(X{1},2);
model.nlf = options.nlf;
model.approx = options.approx;
model.optimiser = options.optimiser;
model.fixInducing = options.fixInducing;
model.sparsePriorType = options.sparsePriorType;
model.IBPisInfinite = options.IBPisInfinite;
model.Opteta = options.Opteta;
model.isVarU = options.isVarU;
model.force_posUpdate = options.force_posUpdate;
model.minJit = 1.;

if isfield(options,'OptMarU'),
    model.OptMarU = options.OptMarU;
end

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

switch model.approx
    case 'ftc'
        % Not Implemented yet
    case {'fitc','pitc', 'dtcvar'}
        if numel(options.numActive) ~= options.nlf
            if numel(options.numActive) == 1,
                options.numActive = options.numActive*ones(1, options.nlf);
            else
                options.numActive = options.numActive(1)*ones(1, options.nlf);
                warning('spmultimodelCreate:noMatchingLF',['The number of latent functions does not match'...
                    ' the number of elements provided in options.numActive']);
            end
        end
        for i = 1: options.nlf,
            numActive = options.numActive(i);
            posX = zeros(numActive, model.ndim);
            switch options.initialInducingPositionMethod
                case 'espaced'
                    % Especially useful for 1D case. It allows to move the
                    % pseudo-inputs a further to the original input range
                    for j = 1:model.ndim,
                        factor = 0.1;
                        med = (max(X{1}(:,j)) - min(X{1}(:,j)))/2;
                        posX(:,j) = linspace(min(X{1}(:,j)) - factor*med, ...
                            max(X{1}(:,j)) + factor*med, numActive)';
                    end
                case 'espacedInRange'
                    % Especially useful for 1D case. It restricts the
                    % initial position of pseudo-inputs to be within the input range
                    for j = 1:model.ndim,
                        posX(:,j) = linspace(min(X{1}(:,j)), max(X{1}(:,j)), numActive)';
                    end
                case 'randomDataIsotopic'
                    % The initial positions of pseudo-inputs are taken from
                    % the data. In the isotopic case, since all inputs all
                    % equal for each output, we use only X(1) to select the
                    % positions, as long as the number of numActive is less
                    % than the size of inputs.
                    if size(X{1},1) >= numActive,
                        totX = cell2mat(X(1));
                    else
                        totX = cell2mat(X');
                    end
                    index = randperm(size(totX,1));
                    posX = totX(index(1:numActive),:);
                case 'randomDataHeterotopic'
                    totX = cell2mat(X'); %TODO check X form
                    index = randperm(size(totX,1));
                    posX = totX(index(1:numActive),:);
                case 'randomComplete'
                    posX = 0.5*rand(numActive, model.ndim);
                case 'fixIndices'
                    posX = X{1}(options.fixIndices,:);
                case 'kmeansIsotopic'
                    posX = kmeanlbg(X{1},numActive);
                case 'kmeansHeterotopic'
                    totX = cell2mat(X);
                    posX = kmeanlbg(totX, numActive);
                otherwise
                    error('This is not valid initialization method for the input variables');
            end
            model.latX{i} = posX;
        end
        %         for i = 1:length(y)
        %             model.outX{i} = X{i}; %TODO check this for multivariate input
        %             model.y{i} = y{i};
        %         end
    otherwise
        error('Unknown model approximation')
end

% Substract the mean
model.y = y;
model.outX = X;
model.m = cell(model.nout,1);
model.md2 = zeros(model.nout,1);
for d = 1:model.nout,
    model.m{d} = model.y{d};
    if model.bias(d)~=0.,
        model.m{d} = model.m{d} - model.bias(d);
    end
    if model.scale(d)~=1.,
        model.m{d} = model.m{d}/model.scale(d);
    end
    model.md2(d) = sum(model.m{d}.^2);
end

model.tieIndices = options.tieOptions.tieIndices;
model.includeNoise = options.includeNoise;

% Set the dimension of each output in model, which will be useful for
% computing sizes and things like that[param, names] = spmultimodelExtractParam(model)

model.indXu = [];
for q = 1:model.nlf,
    model.indXu = [model.indXu q*ones(1, size(model.latX{q},1))];
    model.sizeXu(q) = size(model.latX{q},1);
end

model.indX = [];
for d = 1:model.nout,
    model.indX = [model.indX d*ones(1, size(model.outX{d},1))];
end

for d = 1:model.nout,
    model.sizeX(d) = size(X{d},1);
end

model.k = options.numActive(1);
%%%%%%%%%%%%%%%%%%%%
if isfield(options, 'alpha')
    model.alpha = options.alpha; % Concentration parameter for the IBP prior.
else
    warning('You need to provide a concentration parameter for the IBP prior');
    model.alpha = 1;
end
%%%%%%%%%%%%%%%%%%%%
% model.template = zeros(sum(model.sizeX), model.nout);
localX = cell(1+model.nout,1);
localX{1,1} = model.latX{1};
% startVal = 1;
% endVal = 0;
for j = 1:model.nout,
    %    endVal = endVal + model.sizeX(j);
    %    model.template(startVal:endVal, j) = ones(model.sizeX(k),1);
    localX{1+j,1} = model.outX{j}(:);
    %    startVal = endVal + 1;
end
kern.type = [options.kernType 'global'];
kern.inputDimension = model.ndim;
kern.options = options.kern;
kern.options.nlf = options.nlf;
kern.options.nout = model.nout;
kern.options.approx = model.approx;

if strcmp('lfm', options.kernType),
    kern.options.incMass = false;
end

if isfield(options,'PosS') && options.PosS,
    kern.options.PosS = options.PosS;
end

%kern.options.order = options.kern.order;
kern = kernParamInit(kern);
kernType = multigpKernComposer(options.kernType, 1, 1, model.approx, 1, options);
kern.template.latent = kernCreate(localX{1}, kernType{2});
kern.template.output = kernCreate(localX{2}, kernType{3});
kern.funcNames.computeLat = str2func([kernType{2}{3}  'KernCompute']);
kern.funcNames.computeOut = str2func([kernType{3}{3}  'KernDiagCompute']);
kern.funcNames.computeCross = str2func([kernType{3}{3} 'X' kernType{2}{3} 'KernCompute']);
kern.funcNames.gradientLat = str2func([kernType{2}{3} 'KernGradient']);
kern.funcNames.gradientOut = str2func([kernType{3}{3} 'KernDiagGradient']);
kern.funcNames.gradientCross = str2func([kernType{3}{3} 'X' kernType{2}{3} 'KernGradient']);
kern.funcNames.extractLat = str2func([kernType{2}{3} 'KernExtractParam']);
kern.funcNames.extractOut = str2func([kernType{3}{3} 'KernExtractParam']);
model.kern = kern;
model.kern.isVarS = options.isVarS;
model.kernType = kern.type;
model.kern.paramGroups = speye(model.kern.nParams);
numParams = model.kern.nParams;

% To include independent kernel if model.includeInd
model.includeInd = options.includeInd;
if options.includeInd
    for i=1:model.nout,
        model.kernInd.comp{i} = kernCreate(model.indX{i}, 'rbf');
        numParams = numParams + model.kernInd.comp{i}.nParams;
    end
    model.kernInd.nParams = numParams - model.kern.nParams;
end

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

% Create noise model
switch model.approx,
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

% Initialization of parameters in the gamma priors. We could try to tie
% params adq and bdq
model.gammaPrior = options.gammaPrior;
if options.gammaPrior,
    model.adq = randi(5, model.nout, model.nlf);
    model.adqvec = (model.adq(:))';
    model.adqTransform = optimiDefaultConstraint('positive');
    model.nParams = model.nParams + model.nout*model.nlf;
    model.bdq = randi(5, model.nout, model.nlf);
    model.bdqvec = (model.bdq(:))';
    model.bdqTransform = optimiDefaultConstraint('positive');
    model.nParams = model.nParams + model.nout*model.nlf;
end

% Initialization eta paramters when Opteta flag is active
model.Opteta = options.Opteta;
if options.Opteta,
    model.etadq = .5*ones(model.nout, model.nlf);
    model.etadqvec = reshape(model.etadq,1,model.nout*model.nlf);
    model.etadqTransform = 'logit';
    model.nParams = model.nParams + model.nout*model.nlf;
end

if isfield(options,'UseMeanConstants') && options.UseMeanConstants,
    model.UseMeanConstants = 1;
    model.mu = zeros(model.nout,1);
    model.nParams = model.nParams + model.nout;
else
    model.UseMeanConstants = 0;
end

params = modelExtractParam(model);
model = modelExpandParam(model, params);