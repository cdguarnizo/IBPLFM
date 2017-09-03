function model = ibpmultigpMomentEuast(model)

%Euast = model.Euast;
%Kuuast = model.Kuuast;
%logDetKuuast = model.logDetKuuast; 
%LBt = ibpmultigpLowerBound(model);
for q = 1:model.nlf,
    % Update Kuuast
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
            % Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.Kfu{d,q}.'*model.Kfu{d,q});
            Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.KuuinvKuf{d,q}*model.KuuinvKuf{d,q}.');
        end
    end
    %model.Kuuast{q} = model.Kuu{q}*((Pqq + model.Kuu{q})\model.Kuu{q});
    %L = jitChol(model.Kuuast{q});
    %model.Kuuast{q} = L.'*L;
    %model.logDetKuuast(q) = 2.*sum(log(diag(L)));
    
    Kuuastinv = Pqq + model.Kuuinv{q};
    L = jitChol(Kuuastinv);
    L = L\eye(size(L));
    model.Kuuast{q} = L*L';
    model.logDetKuuast(q) = 2.*sum(log(diag(L)));
    
    if any(isnan(model.Kuuast{q})) | any(isinf(model.Kuuast{q})),
        error('Nan of Inf in Kuuast')
    end
    
    % Update Euast
    KuqfSy = zeros(model.k, 1);
    for d=1:model.nout,
        
        k=1:model.nlf;
        k(q) = [];
        yhatdq = 0;
        if model.isVarS,
            if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
                for q2 = k,
                    yhatdq = yhatdq + (model.etadq(d,q2)*model.muSdq(d,q2))*...
                        (model.KuuinvKuf{d,q2}.'*model.Euast{q2});
                end
                KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q)*model.muSdq(d,q))*(model.KuuinvKuf{d,q}...
                    *(model.m{d} - yhatdq));
            else
                for q2 = k,
                    yhatdq = yhatdq + model.muSdq(d,q2)*...
                        (model.KuuinvKuf{d,q2}.'*model.Euast{q2});
                end
                KuqfSy = KuqfSy + (model.beta(d)*model.muSdq(d,q))*(model.KuuinvKuf{d,q}...
                    *(model.m{d} - yhatdq));
            end
        else
            for q2 = k,
                yhatdq = yhatdq + model.etadq(d,q2)*(model.KuuinvKuf{d,q2}.'*Euast{q2});
            end
            KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q))*(model.KuuinvKuf{d,q}...
                *(model.m{d} - yhatdq));
        end
    end
    model.Euast{q} = model.Kuuast{q}*KuqfSy;
    
%     LB1 = ibpmultigpLowerBound(model);
%     if LB1 - LBt < 0,
%         model.Euast{q} = Euast{q};
%         model.Kuuast{q} = Kuuast{q};
%         model.logDetKuuast(q) = logDetKuuast(q);
%     else
%         LBt = LB1;
%     end
end

% % Update values related to q(u)
% for q = 1:model.nlf,
%     model.Euuast{q,q} = model.Kuuast{q} + model.Euast{q}*model.Euast{q}';
%     for qp = q+1:model.nlf,
%         model.Euuast{q,qp} = model.Euast{q}*model.Euast{qp}';
%         model.Euuast{qp,q} = model.Euuast{q,qp}';
%     end
% end