function param = ftcmgpExtractParam(model)

% FTCMULTIGPEXTRACTPARAM Extract the parameters of an FTCMULTIGP struct.

% FTCMULTIGP

if nargout > 1
    [paramKern, namesKern] = kernExtractParam(model.kern);    
else
    paramKern = kernExtractParam(model.kern);
end
betaParams = [];
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

param = [paramKern betaParams];

% Fix the value of the parameters

if isfield(model, 'fix')
    for i = 1:length(model.fix)
        param(model.fix(i).index) = model.fix(i).value;
    end
end

if nargout > 1
    names = {namesKern{:}, betaParamNames{:}};
end