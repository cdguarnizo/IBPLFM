function model = ibpmultigpSortModel(model)

%% Sort etadq according to its values
[~, sorteta] = sort(sum(abs(model.etadq.*model.kern.sensitivity)),'descend');
if any(sorteta~=1:model.nlf),
    temp = model.latX;
    for q = 1:model.nlf,
        model.latX{q} = temp{sorteta(q)};
    end
    model.etadq = model.etadq(:, sorteta);
    
    model.tau1 = model.tau1(sorteta);
    model.tau2 = model.tau2(sorteta);
    model.Elog1mProdpim = model.Elog1mProdpim(sorteta);
    model.Elogpiq = model.Elogpiq(sorteta);
    
    if strcmp(model.kern.type(1:2),'gg'),
        model.kern.precisionU = model.kern.precisionU(sorteta);
    else
        model.kern.inverseWidth = model.kern.inverseWidth(sorteta);
    end
    model.kern.sensitivity = model.kern.sensitivity(:, sorteta);
    if isfield(model, 'gamma'),
        model.gamma = model.gamma(sorteta);
    end
    if model.isVarU,
        temp = model.Kuuast;
        temp2 = model.Euast;
        for q = 1:model.nlf,
            model.Kuuast{q} = temp{sorteta(q)};
            model.Euast{q} = temp2{sorteta(q)};
        end
        %model.logDetKuuast = model.logDetKuuast(sorteta);
    end
    %model = ibpmultigpComputeKernels(model);
end