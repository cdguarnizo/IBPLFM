function [umean, uvar] = ibpmultigpPosteriorLatent(model, Xtest)

umean = cell(model.nlf,1);
uvar = cell(model.nlf,1);
Xlattest = cell(model.nlf,1);

for q=1:model.nlf,
    Xlattest{q} = Xtest;
end

if isfield(model, 'gamma') && ~isempty(model.gamma)
    Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX, model.gamma);
else
    Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX);
end

A = cell(model.nlf,1);
umean = cell(model.nlf,1);
uvar = cell(model.nlf,1);
Ainv = cell(model.nlf,1);
for q=1:model.nlf,
    T=0; %\bar{K}_{u_q,u_q}
    Ut = 0;
    if model.isVarS,
        if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
            for d=1:model.nout,
                Ut = Ut + (model.etadq(d,q)*model.muSdq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.m{d});
                T = T + (model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
            end
        else
            for d=1:model.nout,
                Ut = Ut + (model.muSdq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.m{d});
                T = T + ((model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
            end
        end
    else
        for d=1:model.nout,
            Ut = Ut + (model.etadq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.m{d});
            T = T + (model.etadq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
        end
    end
    
    A{q} = model.Kuu{q} + T; % A_{q,q} = K_{u_q,u_q} + \bat{K}_{u_q,u_q} 

    Ainv{q} = pdinv(A{q});
    T = Kusu{q}*Ainv{q};
    umean{q} = T*Ut;
    uvar{q} = diag(T*Kusu{q}');
end