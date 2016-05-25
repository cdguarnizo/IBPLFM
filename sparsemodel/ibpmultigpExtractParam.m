function  [param, names] = ibpmultigpExtractParam(model)

% IBPMULTIGPEXTRACTPARAM Extract the parameters of an IBPMULTIGP struct.

% IBPMULTIGP


if nargout > 1
    [paramKern, namesKern] = kernExtractParam(model.kern);    
else
    paramKern = kernExtractParam(model.kern);
end

if isfield(model, 'meanFunction') && ~isempty(model.meanFunction)
    if nargout>1
        [meanFuncParams, meanFuncParamNames] = meanExtractParam(model.meanFunction);
        for i = 1:length(meanFuncParamNames)
            meanFuncParamNames{i} = ['mean Func ' meanFuncParamNames{i}];
        end
    else
        meanFuncParams = meanExtractParam(model.meanFunction);
    end
else
    meanFuncParamNames = {};
    meanFuncParams =[];
end

if isfield(model, 'gamma') && ~isempty(model.gamma)
    fhandle = str2func([model.gammaTransform 'Transform']);
    gammaParams = fhandle(model.gamma, 'xtoa');    
    if nargout>1
        gammaParamNames = cell(model.nlf,1);
        for i = 1:length(gammaParams)
            gammaParamNames{i} = ['Gamma ' num2str(i)];
        end
    end
else
    gammaParamNames = {};
    gammaParams =[];
end

if isfield(model, 'beta') && ~isempty(model.beta)
    fhandle = str2func([model.betaTransform 'Transform']);
    betaParams = fhandle(model.beta, 'xtoa');    
    if nargout>1
        betaParamNames = cell(model.nout,1);
        for i = 1:length(betaParams)
            betaParamNames{i} = ['Beta ' num2str(i)];
        end
    end
else
    betaParamNames = {};
    betaParams =[];
end

if model.isVarS,
    if isfield(model, 'adq') && ~isempty(model.adq)
        fhandle = str2func([model.adqTransform 'Transform']);
        adqParams = fhandle(model.adqvec, 'xtoa');
        if nargout>1
            adqParamNames = cell(model.nout*model.nlf,1);
            for i = 1:length(adqParamNames)
                adqParamNames{i} = ['adq ' num2str(i)];
            end
        end
    else
        adqParamNames = {};
        adqParams =[];
    end
    
    if isfield(model, 'bdq') && ~isempty(model.bdq)
        fhandle = str2func([model.bdqTransform 'Transform']);
        bdqParams = fhandle(model.bdqvec, 'xtoa');
        if nargout>1
            bdqParamNames = cell(model.nout*model.nlf,1);
            for i = 1:length(bdqParamNames)
                bdqParamNames{i} = ['bdq ' num2str(i)];
            end
        end
    else
        bdqParamNames = {};
        bdqParams =[];
    end
else
    adqParamNames = {};
    adqParams =[];
    bdqParamNames = {};
    bdqParams =[];
end

if isfield(model, 'fixinducing') && ~model.fixinducing,
    %We assume that all latent functions have the same number of inducing
    %points, each point in R^p
    width = size(model.latX{1},1)*size(model.latX{1},2);
    XParams = zeros(1, model.nlf*width);
    XParamNames = cell(model.nlf*width, 1);
    for k=1:model.nlf,
        XParams(1 + (k-1)*width: k*width) = model.latX{k}(:)';
        for l=1:width,
            XParamNames{(k-1)*width + l} = ['X latent ' num2str(k)];
        end
    end
else
    XParamNames = {};
    XParams = [];
end

if model.Opteta,
    if isfield(model, 'etadq') && ~isempty(model.etadq),
        fhandle = str2func([model.etadqTransform 'Transform']);
        etadqParams = fhandle(model.etadqvec, 'xtoa');
        if nargout>1,
            etadqParamNames = cell(model.nout*model.nlf,1);
            for i = 1:length(etadqParamNames),
                etadqParamNames{i} = ['etadq ' num2str(i)];
            end
        end
    else
        etadqParamNames = {};
        etadqParams =[];
    end
else
    etadqParamNames = {};
    etadqParams =[];
end

param = [paramKern meanFuncParams gammaParams betaParams adqParams bdqParams XParams etadqParams];

% Fix the value of the parameters

if isfield(model, 'fix')
    for i = 1:length(model.fix)
        param(model.fix(i).index) = model.fix(i).value;
    end
end

if nargout > 1
    names = {namesKern{:}, meanFuncParamNames{:}, gammaParamNames{:}, ...
        betaParamNames{:}, adqParamNames{:}, bdqParamNames{:}, XParamNames{:}, ...
        etadqParamNames{:}};
end