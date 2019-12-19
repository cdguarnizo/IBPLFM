function model = ibpmultigpUpdateA(model)

% IBPMULTIGPUPDATEA

% IBPMULTIGP

% Compute Kfutilde = E[ZdqSdq]*Kfu
model.Kfutilde = cell(model.nout, model.nlf);
for q=1:model.nlf
    for d=1:model.nout
        model.Kfutilde{d, q} = model.EZdqSdq(d,q)*model.Kfu{d,q};
    end
end


% Compute Kuuinv
for q=1:model.nlf,
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        [model.Kuuinv{q}, model.sqrtKuu{q}, ~, model.sqrtKuuinv{q}] = pdinv(model.KuuGamma{q});
        model.logDetKuu(q) = logdet(model.KuuGamma{q}, model.sqrtKuu{q});
    else
        [model.Kuuinv{q}, model.sqrtKuu{q}, ~, model.sqrtKuuinv{q}] = pdinv(model.Kuu{q});
        model.logDetKuu(q) = logdet(model.Kuu{q}, model.sqrtKuu{q});
    end
end

% Compute A and P
for q1 = 1:model.nlf
    KuftSigmaKuft = zeros(model.k);
    for d = 1:model.nout
        const = model.beta(d)*model.EZdqS2dq(d, q1);
        KufKfu = model.Kfu{d, q1}.'*model.Kfu{d, q1}; 
        KuftSigmaKuft = KuftSigmaKuft + const*KufKfu; 
    end
    sqrtKuftSigmaKuft = jitChol(KuftSigmaKuft + 1e-8*eye(model.k));
    %sqrtKuftSigmaKuft = jitChol(KuftSigmaKuft);
    KuuinvsqrtKuftSigmaKuft = model.Kuuinv{q1}*sqrtKuftSigmaKuft.';
    model.P{q1, q1} = KuuinvsqrtKuftSigmaKuft*KuuinvsqrtKuftSigmaKuft.';
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        model.A{q1} = model.KuuGamma{q1} + KuftSigmaKuft;
    else
        model.A{q1} = model.Kuu{q1} + KuftSigmaKuft;
    end
    model.Ainv{q1} = pdinv(model.A{q1}); 
    for q2 = 1:q1-1
        KuftSigmaKuft = zeros(model.k);
        for d = 1:model.nout
           const = model.beta(d)*model.EZdqSdq(d, q1)*model.EZdqSdq(d, q2);
           KufKfu = model.Kfu{d, q1}.'*model.Kfu{d, q2};
           KuftSigmaKuft = KuftSigmaKuft + const*KufKfu;
        end        
        model.P{q1, q2} = model.Kuuinv{q1}*KuftSigmaKuft*model.Kuuinv{q2};
        model.P{q2, q1} = model.P{q1, q2}';
    end
end

% Compute mq
for q =1: model.nlf,    
    mqv = zeros(model.k, 1);
    for d =1: model.nout,
        model.KfusqrtKuuinv{d,q} = model.Kfu{d,q}*model.sqrtKuuinv{q};        
        mqv = mqv + model.beta(d)*(model.Kfutilde{d,q}.'*model.m{d});        
    end
    model.mq{q, 1} = model.Kuuinv{q}*mqv;
end

model.cdq = zeros(model.nout, model.nlf);
model.KfdgivenUq = zeros(model.nout, model.nlf);
for d = 1:model.nout
    for q = 1:model.nlf
        model.KfdgivenUq(d,q) = sum(model.Kff{d,q}) - sum(sum(...
            model.KfusqrtKuuinv{d,q}.*model.KfusqrtKuuinv{d,q}));
        model.cdq(d,q) = model.beta(d)*model.KfdgivenUq(d,q);
    end
end










