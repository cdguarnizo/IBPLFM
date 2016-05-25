function model = ibpmultigpComputeKernels(model)

% IBPMULTIGPCOMPUTEKERNELS

% IBPMULTIGP

fhandle = str2func([model.kernType 'KernCompute']);
if isfield(model, 'gamma') && ~isempty(model.gamma)
    [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
        model.outX, model.latX, model.gamma);
else
    [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
        model.outX, model.latX);
end

% Compute Kuuinv
for q = 1:model.nlf,
%     [model.Kuuinv{q}, model.sqrtKuu{q}, ~, model.sqrtKuuinv{q}] = pdinv(model.Kuu{q});
%     model.Kuu{q} = model.sqrtKuu{q}.'*model.sqrtKuu{q}; %Rebuild of Kuu
%     model.logDetKuu(q) = logdet(model.Kuu{q}, model.sqrtKuu{q});
    
    model.Kuuinv{q} = model.Kuu{q}\eye(size(model.Kuu{q}));
    model.logDetKuu(q) = logdet(model.Kuu{q});
end

% Compute cdq
model.cdq = zeros(model.nout,model.nlf);
for d = 1:model.nout,
    for q=1:model.nlf,
        %model.cdq(d,q) = model.beta(d)*(sum(model.Kff{d,q}) - ...
        %    trace(model.Kfu{d,q}*(model.Kuu{q}\model.Kfu{d,q}.')));
        
        %model.KuuinvKuf{d,q} = model.Kuuinv{q}*model.Kfu{d,q}.';
        model.KuuinvKuf{d,q} = model.Kuu{q}\model.Kfu{d,q}.';
        
        model.cdq(d,q) = model.beta(d)*(sum(model.Kff{d,q}) - ...
            trace(model.Kfu{d,q}*(model.KuuinvKuf{d,q})));
        
        if isnan(model.cdq(d,q)),
            error('cdq is NaN')
        end
        
    end
end

if any(model.cdq<0),
    warning('Negative values in cdq')
end

%TODO: Talk this error with Mauricio
%model.cdq(model.cdq < 0.) = 0.;

if ~model.isVarU,
    model.P = 0;
    model.m2 = 0;
    for d = 1:model.nout,
        indd = d*ones(1,length(model.m{d}));
        EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2)...
            + diag(model.etadq(d,:));
        EZ = model.etadq(indd,model.indXu);
        
        Kfdu = cell2mat( model.Kfu(d,:) );
        model.P = model.P + model.beta(d)*(EZ2(model.indXu,model.indXu).*(Kfdu.'*Kfdu));
        model.m2 = model.m2 + model.beta(d)*((EZ .* Kfdu).'*model.m{d});
    end
    model.A = blkdiag(model.Kuu{:}) + model.P;
    model.Ainv = model.A\eye(size(model.A));
    model.logDetA = logdet(model.A);
    
    %[model.Ainv, sqrtA, ~, sqrtAinv] = pdinv(model.A);
    %model.logDetA = logdet(model.A, sqrtA);
    %model.sqrtAinvm = sqrtAinv*model.m2;
    
%     %% As Mauricio did
%     model.logDetDT = 0;
%     model.traceDinvyy = 0;
%     model.KtildeT = 0;
%     
%     for r =1: model.nlf,
%         for k =1: model.nout,
%             model.KuuinvKuy{r,k} = model.Kuuinv{r}*model.Kfu{k,r}';
%         end
%     end
%     
%     for k =1:model.nout
%         model.D{k} = 1/model.beta(k)*ones(size(model.outX{k},1),1);
%         model.Dinv{k} = model.beta(k)*ones(size(model.outX{k},1),1);
%         model.logDetD{k} = -size(model.outX{k},1)*log(model.beta(k));
%         for r=1:model.nlf
%             model.KuyDinv{r,k} = model.beta(k)*model.Kfu{k,r}';
%         end
%         KyuKuuinvKuy = zeros(size(model.Kff{k},1),1);
%         for r =1: model.nlf,
%             KyuKuuinvKuy = KyuKuuinvKuy + sum(model.Kfu{k,r}.*model.KuuinvKuy{r,k}',2); %TODO change this
%         end
%         model.Ktilde{k} = model.Kff{k} - KyuKuuinvKuy;
%         temp = model.beta(k)*sum(model.Ktilde{k});
%         model.KtildeT = model.KtildeT + temp;
%         model.logDetDT = model.logDetDT + model.logDetD{k};
%         model.traceDinvyy = model.traceDinvyy + sum(model.Dinv{k}.*model.m{k}.*model.m{k});
%     end
%     
%     for r = 1:model.nlf,
%         model.KuyDinvy{r,1} = zeros(model.k,1);
%         for q =1:model.nout,
%             model.KuyDinvy{r} = model.KuyDinvy{r} + model.KuyDinv{r,q}*model.m{q};
%         end
%     end
%     
%     for k =1:model.nlf,
%         KuyDinvKyu = zeros(model.k);
%         for q =1:model.nout,
%             KuyDinvKyu = KuyDinvKyu + model.KuyDinv{k,q}*model.Kfu{q,k};
%         end
%         model.A{k,k} = model.Kuu{k} + KuyDinvKyu;
%         for r = 1:k-1,
%             KuyDinvKyu = zeros(model.k, model.k);
%             for q = 1:model.nout,
%                 KuyDinvKyu = KuyDinvKyu + model.KuyDinv{k,q}*model.Kfu{q,r};
%             end
%             model.A{k,r} = KuyDinvKyu;
%             model.A{r,k} = KuyDinvKyu';
%         end
%     end
%     
%     A = cell2mat(model.A);
%     KuyDinvy = cell2mat(model.KuyDinvy);
%     [Ainv, sqrtA]  = pdinv(A);
%     sqrtAinv = jitChol(Ainv);
%     model.logDetA = logdet(A, sqrtA);
%     
%     model.Ainv  = mat2cell(Ainv, model.k*ones(1,model.nlf), model.k*ones(1,model.nlf));
%     model.sqrtA = mat2cell(sqrtA,model.k*ones(1,model.nlf), model.k*ones(1,model.nlf));
%     model.sqrtAinvKuyDinvy = mat2cell(sqrtAinv*KuyDinvy, model.k*ones(1,model.nlf), 1);
%     
%     for r = 1:model.nlf,
%         model.AinvKuyDinvy{r,1} = zeros(model.k,1);
%         for k = 1:model.nlf,
%             model.AinvKuyDinvy{r} = model.AinvKuyDinvy{r} + model.Ainv{r,k}*model.KuyDinvy{k};
%         end
%     end
    
end