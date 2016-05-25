function model = ibpmultigpMomentEuast(model)

%Save old Euast
Euast = model.Euast;
for q = 1:model.nlf
    %Update Kuuast
    model.Kuuast{q} = 0;
    Pqq = 0;
    if model.isVarS,
        if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp'),
            for d=1:model.nout,
                Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))...
                    *(model.Kfu{d,q}.'*model.Kfu{d,q});
            end
        else
            for d=1:model.nout,
                Pqq = Pqq + (model.beta(d)*(model.muSdq(d,q)^2+model.varSdq(d,q)))...
                    *(model.Kfu{d,q}.'*model.Kfu{d,q});
            end
        end
    else
        for d=1:model.nout,
            Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.Kfu{d,q}.'*model.Kfu{d,q});
        end
    end
    %PqqKuuinv = Pqq*model.Kuuinv{q};
    %model.Kuuast{q} = (PqqKuuinv + eye(size(Pqq)))\model.Kuu{q};
    
    model.Kuuast{q} = model.Kuu{q}*((Pqq + model.Kuu{q})\model.Kuu{q});
    
    %Update Euast
    KuqfSy = zeros(model.k, 1);
    for d=1:model.nout,
        
        k=1:model.nlf;
        k(q) = [];
        yhatdq = 0;
        if model.isVarS,
            if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
                for q2 = k,
                    yhatdq = yhatdq + (model.etadq(d,q2)*model.muSdq(d,q2))*...
                        (model.KuuinvKuf{d,q2}'*Euast{q2});
                    %yhatdq = yhatdq + (model.etadq(d,q2)*model.muSdq(d,q2))*...
                    %    (model.Kfu{d,q2}/model.Kuu{q2})*Euast{q2};
                end
                KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q)*model.muSdq(d,q))*(model.Kfu{d,q}'...
                    *(model.m{d} - yhatdq));
            else
                for q2 = k,
                    yhatdq = yhatdq + model.muSdq(d,q2)*...
                        (model.KuuinvKuf{d,q2}*Euast{q2});
                    %yhatdq = yhatdq + model.muSdq(d,q2)*...
                    %    (model.Kfu{d,q2}/model.Kuu{q2})*Euast{q2};
                end
                KuqfSy = KuqfSy + (model.beta(d)*model.muSdq(d,q))*(model.Kfu{d,q}.'...
                    *(model.m{d} - yhatdq));
            end
        else
            for q2 = k,
                yhatdq = yhatdq + model.etadq(d,q2)*(model.Kfu{d,q2}*(model.Kuuinv{q2}*Euast{q2}));
                %yhatdq = yhatdq + model.etadq(d,q2)*((model.Kfu{d,q2}/model.Kuu{q2})*Euast{q2});
            end
            KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q))*(model.Kfu{d,q}'...
                *(model.m{d} - yhatdq));
        end
    end
    %model.Euast{q} = (PqqKuuinv + eye( size(Pqq) ))\KuqfSy;
    model.Euast{q} = model.Kuuast{q}*(model.Kuu{q}\KuqfSy);
end