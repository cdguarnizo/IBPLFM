function model = ibpmultigpUpdateA(model)

% IBPMULTIGPUPDATEA

% IBPMULTIGP

% % Compute Kfutilde = E[ZdqSdq]*Kfu
% model.Kfutilde = cell(model.nout, model.nlf);
% for q=1:model.nlf
%     for d=1:model.nout
%         %model.Kfutilde{d, q} = model.EZdqSdq(d,q)*model.Kfu{d,q};
%         model.Kfutilde{d, q} = (model.muSdq(d,q)*model.etadq(d,q))*model.Kfu{d,q}; %CGL
%     end
% end

% Compute Kuuinv
for q = 1:model.nlf,
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        [model.Kuuinv{q}, model.sqrtKuu{q}, ~, model.sqrtKuuinv{q}] = pdinv(model.KuuGamma{q});
        model.logDetKuu(q) = logdet(model.KuuGamma{q}, model.sqrtKuu{q});
    else
        [model.Kuuinv{q}, model.sqrtKuu{q}, ~, model.sqrtKuuinv{q}] = pdinv(model.Kuu{q});
        model.logDetKuu(q) = logdet(model.Kuu{q}, model.sqrtKuu{q});
    end
end
% Compute cdq
model.cdq = zeros(model.nout,model.nlf);
for d = 1:model.nout,
    for q=1:model.nlf,
        model.cdq(d,q) = model.beta(d)*(sum(model.Kff{d,q}) - ...
            trace(model.Kfu{d,q}*model.Kuuinv{q}*model.Kfu{d,q}.'));
    end
end

%TODO: Talk this error with Mauricio
model.cdq(model.cdq < 0) = 0.;