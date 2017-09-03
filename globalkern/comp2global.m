%From compound to global kern params
function params = comp2global(model)
param = modelExtractParam(model);
if strcmp(model.kernType,'lfm')
    invwidth = paramNameRegularExpressionLookup(model, 'rbf .* inverse width');
    mass = paramNameRegularExpressionLookup(model, 'multi 1 lfm .* mass');
    spring = paramNameRegularExpressionLookup(model, 'multi 1 lfm .* spring');
    damper = paramNameRegularExpressionLookup(model, 'multi 1 lfm .* damper');
    params = exp(param([invwidth mass spring damper]));
end

if strcmp(model.kernType,'gg')
    precG = paramNameRegularExpressionLookup(model, ['multi 1 gg .* inverse width output']);
    precU = paramNameRegularExpressionLookup(model, '.* inverse width latent');
    params = exp(param([precU precG]));
end