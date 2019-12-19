function f = ibpmultigpLowerBoundOptimise(model)

% IBPMULTIGPLOWERBOUND FOR IBP Parameters

% IBPMULTIGP
f = 0;

if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
    EZS2 = model.etadq.*(model.varSdq + model.muSdq.^2);
    EZS = model.etadq.*model.muSdq;
else
    EZS2 = model.varSdq + model.muSdq.^2;
    EZS = model.muSdq;
end

% Add trace \sum mq E[uq]
for q = 1:model.nlf
    mq=0;
    for d = 1:model.nout
        mq = mq + EZS(d,q)*model.beta(d)*model.Kfu{d,q}.'*model.m{d};
    end
    mq = model.Kuuinv{q}*mq;
    f = f + trace(mq*model.Euast{q}.');
end
% Add trace \sum \sum Pqq' E[uq'uq]
for q1 = 1:model.nlf
    Pqq = 0;
    for d = 1:model.nout
        Pqq = Pqq + EZS2(d,q1)*...
            model.beta(d)*model.Kfu{d,q1}.'*model.Kfu{d,q1};
    end
    Pqq = model.Kuuinv{q1}*Pqq*model.Kuuinv{q1};
    
    f = f - 0.5*trace(Pqq*(model.Kuuast{q1} + model.Euast{q1}*model.Euast{q1}.'));
    k = 1:model.nlf;
    k(q1) = [];
    for q2 = k
        Pqqp = 0;
        for d = 1:model.nout
            Pqqp = Pqqp + EZS(d,q1)*EZS(d,q2)*...
                model.beta(d)*model.Kfu{d,q1}.'*model.Kfu{d,q2};
        end
        Pqqp = model.Kuuinv{q1}*Pqqp*model.Kuuinv{q2};
        f = f - 0.5*trace(Pqqp*(model.Euast{q2}*model.Euast{q1}.'));
    end
end
% Add \sum Kuquqinv Euuast
for q=1:model.nlf
   f = f - 0.5*trace(model.Kuuinv{q}*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}.')); 
   %f = f - 0.5*trace(model.Kuuinv{q}*model.Euuast{q,q}); 
end
% Add logdet Kuu
f = f - 0.5*sum(model.logDetKuu);
for d=1:model.nout
    % Add 0.5 log Sigma_w
    f = f + 0.5*model.sizeX(d)*log(model.beta(d)); 
    % Add trace (Sigma_w yy^{\top})
    f = f - 0.5*model.beta(d)*sum(model.m{d}.^2);
end
f = f - 0.5*sum(sum( model.cdq.*EZS2 ));
if model.gammaPrior
    % Add \sum \sum log Gamma(adq)
    f = f - sum(sum(log(gamma(model.adq))));
    % Add \sum \sum adq log bdq
    f = f + sum(sum( model.adq.*log(model.bdq)));
    % Add \sum \sum (adq - 1)[psi(adqast) - log (bdqast)]
    f = f + sum(sum( (model.adq - 1).*(psi(model.adqast) -log(model.bdqast))));
    % Add bdq (adqast/bdqast)
    f = f - sum(sum(model.bdq.*(model.adqast./model.bdqast)));
end