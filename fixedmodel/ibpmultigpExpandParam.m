function model = ibpmultigpExpandParam(model, params)

% IBPMULTIGPEXPANDPARAM Expand the parameters into an IBPMULTIGP struct.

% IBPMULTIGP

paramPart = real(params);

if isfield(model, 'fix')
    for i = 1:length(model.fix)
       paramPart(model.fix(i).index) = model.fix(i).value;
    end
end

startVal = 1;
endVal = model.kern.nParams;
kernParams = paramPart(startVal:endVal);
%kernParams(1:model.nlf) = sort(kernParams(1:model.nlf));
if length(kernParams) ~= model.kern.nParams
    error('kern parameter vector is incorrect length');
end

model.kern = kernExpandParam(model.kern, kernParams);

% Check if there is a mean function.
if isfield(model, 'meanFunction') && ~isempty(model.meanFunction)
    startVal = endVal + 1;
    endVal = endVal + model.meanFunction.nParams;
    model.meanFunction = meanExpandParam(model.meanFunction, ...
        paramPart(startVal:endVal));
end

% Check if there is a gamma parameter.
if isfield(model, 'gamma') && ~isempty(model.gamma)
    startVal = endVal + 1;
    endVal = endVal + model.nlf;
    fhandle = str2func([model.gammaTransform 'Transform']);
    model.gamma = fhandle(paramPart(startVal:endVal), 'atox');
end

% Check if there is a beta parameter.
if isfield(model, 'beta') && ~isempty(model.beta)
    startVal = endVal + 1;
    endVal = endVal + model.nout;
    model.beta(model.beta > 1e4) = 1e4;
    fhandle = str2func([model.betaTransform 'Transform']);
    model.beta = fhandle(paramPart(startVal:endVal), 'atox');
end

if model.isVarS,
    % Check if there are adq parameters.
    if model.gammaPrior && isfield(model, 'adq') && ~isempty(model.adq)
        startVal = endVal + 1;
        endVal = endVal + model.nout*model.nlf;
        fhandle = str2func([model.adqTransform 'Transform']);
        model.adqvec = fhandle(paramPart(startVal:endVal), 'atox');
        model.adq = reshape(model.adqvec, model.nout, model.nlf);
    end
    
    % Check if there are bdq parameters.
    if model.gammaPrior && isfield(model, 'bdq') && ~isempty(model.bdq)
        startVal = endVal + 1;
        endVal = endVal + model.nout*model.nlf;
        fhandle = str2func([model.bdqTransform 'Transform']);
        model.bdqvec = fhandle(paramPart(startVal:endVal), 'atox');
        model.bdq = reshape(model.bdqvec, model.nout, model.nlf);
    end
end
% Check if latent input points are optimized
if isfield(model, 'fixInducing') && ~model.fixInducing,
    for k=1:model.nlf,
        startVal = endVal + 1;
        endVal = endVal + length(model.latX{k}(:));
        model.latX{k} = reshape(paramPart(startVal:endVal), ...
            size(model.latX{k},1), size(model.latX{k},2));
    end
end

if isfield(model, 'Opteta') && model.Opteta,
    startVal = endVal + 1;
    endVal = endVal + model.nout*model.nlf;
    fhandle = str2func([model.etadqTransform 'Transform']);
    model.etadqvec = fhandle(paramPart(startVal:endVal), 'atox');
    model.etadq = reshape(model.etadqvec, model.nout, model.nlf);
end

if isfield(model, 'UseMeanConstants') && model.UseMeanConstants,
    startVal = endVal + 1;
    endVal = endVal + model.nout;
    model.mu = reshape(paramPart(startVal:endVal),model.nout,1);
end