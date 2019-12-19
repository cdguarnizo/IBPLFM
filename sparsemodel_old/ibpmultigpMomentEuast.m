function model = ibpmultigpMomentEuast(model)

%Save old Euast
Euast = model.Euast;

for q=1:model.nlf
    %Update Kuuast
    model.Kuuast{q} = 0;
    Pqq = 0;
    if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
        for d = 1:model.nout
            Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))...
                *model.Kfu{d,q}.'*model.Kfu{d,q};
        end
    else
        for d = 1:model.nout
            Pqq = Pqq + model.beta(d)*(model.muSdq(d,q)^2+model.varSdq(d,q))...
                *model.Kfu{d,q}.'*model.Kfu{d,q};
        end
    end
    
    Pqq = model.Kuuinv{q}*Pqq*model.Kuuinv{q};
    model.Kuuast{q} = pdinv(Pqq + model.Kuuinv{q});
    
    %Update Euast
    KuqfSy = zeros(model.k, 1);
    for d = 1:model.nout
        k = 1:model.nlf;
        k(q) = [];
        yhatdq = 0;
        if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
            for q2 = k
                yhatdq = yhatdq + model.etadq(d,q2)*model.muSdq(d,q2)*...
                    model.Kfu{d,q2}*model.Kuuinv{q2}*Euast{q2};
            end
            KuqfSy = KuqfSy + model.beta(d)*(model.etadq(d,q)*model.muSdq(d,q)*model.Kfu{d,q}.'...
                *(model.m{d} - yhatdq));
        else
            if model.actyhat
                for q2 = k
                    yhatdq = yhatdq + model.muSdq(d,q2)*...
                        model.Kfu{d,q2}*model.Kuuinv{q2}*Euast{q2};
                end
                KuqfSy = KuqfSy + model.beta(d)*(model.muSdq(d,q)*model.Kfu{d,q}.'...
                    *(model.m{d} - yhatdq));
            else
                KuqfSy = KuqfSy + model.beta(d)*(model.muSdq(d,q)*model.Kfu{d,q}.'...
                    *model.m{d});
            end
        end
    end
    model.Euast{q} = model.Kuuast{q}*(model.Kuuinv{q}*KuqfSy);
end