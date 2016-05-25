function [dKff, dKfu, dKuu, dKSigma] = ibpmultigpLowerBoundGradCovMat(model)

% IBPMULTIGPLOWERBOUNDGRADCOVMAT

% IBPMULTIGP

%COPYRIGHT : Cristian Guarnizo, 2014

if model.isVarU,
    dKfu = cell(model.nout,model.nlf);
    dKuu = cell(1,model.nlf);
    dKSigma = zeros(1,model.nout);
    dKff = cell(model.nout,model.nlf);
    if model.isVarS,
        if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
            EZS2 = model.etadq.*(model.varSdq + model.muSdq.^2);
            EZdqSdq = model.etadq.*model.muSdq;
        else
            EZS2 = model.varSdq + model.muSdq.^2;
            EZdqSdq = model.muSdq;
        end
    else
        if strcmp(model.sparsePriorType,'ard'),
            EZS2 = ones(size(model.muSdq));
            EZdqSdq = EZS2;
        else
            EZS2 = model.etadq;
            EZdqSdq = model.etadq;
        end
    end
    %Helpers to evaluate the gradient of Sigma_d (noise related matrix)
    T1Sigmad = cell(model.nout,1);
    T2Sigmad = cell(model.nout,1);
    T5Sigmad = cell(model.nout,1);
    
    for q = 1:model.nlf,
        T1Kuu = 0;
        T3Kuu = 0;
        T4Kuu = 0;
        Euquq = (model.Kuuast{q} + model.Euast{q}*model.Euast{q}.');
        for d=1:model.nout,
            if q==1,
                T1Sigmad{d} = 0;
                T2Sigmad{d} = 0;
                T5Sigmad{d} = 0;
            end
            
            T1 = EZdqSdq(d,q)*model.beta(d)*model.m{d}*model.Euast{q}.';
            T1Kuu = T1Kuu + model.Kfu{d,q}.'*T1;
            
            T2 = 0;
            if model.nlf ~= 1,
                k = 1:model.nlf;
                k(q) = [];
                for q1 = k,
                    T2 = T2 + EZdqSdq(d,q1)*(model.KuuinvKuf{d,q1}.'*...
                        (model.Euast{q1}*model.Euast{q}.'));
                    %T2 = T2 + EZdqSdq(d,q1)*(model.Kfu{d,q1}*(model.Kuu{q1}\...
                    %    (model.Euast{q1}*model.Euast{q}.')));
                end
                T2s = EZdqSdq(d,q)*T2;
                T2 = model.beta(d)*T2s;
            end
            KuuinvEuquq = model.Kuu{q}\Euquq;
            
%             T3 = EZS2(d,q)*(model.beta(d)*model.Kfu{d,q})*(eye(size(Euquq,1)) - model.Kuuinv{q}*Euquq);
%             dKfu{d,q} = (T1 - T2 + T3)*model.Kuuinv{q};
            
            T3 = EZS2(d,q)*(model.beta(d)*model.Kfu{d,q})*(eye(size(Euquq,1)) - KuuinvEuquq);
            dKfu{d,q} = (T1 - T2 + T3)*model.Kuuinv{q};
            
            T3Kuu = T3Kuu + model.Kfu{d,q}.'*T2;
            T4Kuu = T4Kuu + EZS2(d,q)*model.beta(d)*model.Kfu{d,q}.'*model.Kfu{d,q};
            
            T1Sigmad{d} = T1Sigmad{d} + model.m{d}*model.Euast{q}.'*model.Kuuinv{q}*EZdqSdq(d,q)...
                *model.Kfu{d,q}.';
            
            %T2Sigmad{d} = T2Sigmad{d} + (T2s + EZS2(d,q)*model.Kfu{d,q}*model.Kuuinv{q}*Euquq)*...
            %    model.Kuuinv{q}*model.Kfu{d,q}.';
            
%             T2Sigmad{d} = T2Sigmad{d} + (T2s + EZS2(d,q)*model.Kfu{d,q}*model.KuuinvEuquq)*...
%                 model.Kuuinv{q}*model.Kfu{d,q}.';

            T2Sigmad{d} = T2Sigmad{d} + (T2s + EZS2(d,q)*model.Kfu{d,q}*KuuinvEuquq)*...
                model.KuuinvKuf{d,q};
            
%             T5Sigmad{d} = T5Sigmad{d} + EZS2(d,q)*(sum(model.Kff{d,q}) - trace(model.Kfu{d,q}...
%                 *model.Kuuinv{q}*model.Kfu{d,q}.'));
            
            T5Sigmad{d} = T5Sigmad{d} + EZS2(d,q)*(sum(model.Kff{d,q}) - trace(model.Kfu{d,q}...
                *model.KuuinvKuf{d,q}));

            if q == model.nlf,
                dKSigma(d) = trace( T1Sigmad{d} - 0.5*T2Sigmad{d}) - 0.5*sum(model.m{d}.^2)...
                    - 0.5*T5Sigmad{d} + 0.5*model.sizeX(d)/model.beta(d);
            end
            dKff{d,q} = -0.5*EZS2(d,q)*model.beta(d)*eye(length(model.m{d}));
        end
        %T4Kuu = T4Kuu*model.Kuuinv{q}*Euquq + Euquq*model.Kuuinv{q}*T4Kuu - T4Kuu;
        T4Kuu = T4Kuu*KuuinvEuquq + KuuinvEuquq'*T4Kuu - T4Kuu;
        %dKuu{q} = model.Kuuinv{q}*(-T1Kuu + .5*Euquq + T3Kuu + .5*T4Kuu)*model.Kuuinv{q} - .5*model.Kuuinv{q};
        dKuu{q} = (model.Kuu{q}\(-T1Kuu + .5*Euquq + T3Kuu + .5*T4Kuu))*model.Kuuinv{q} - .5*model.Kuuinv{q};
        dKuu{q} = (dKuu{q}.' + dKuu{q})/2;
        %dKuu{q} = dKuu{q}.' + dKuu{q} - diag(diag(dKuu{q}));
    end
else %Only for isVarS = false
    dKfu = cell(model.nout);
    dKff = cell(model.nout,model.nlf);
    dKSigma = zeros(1,model.nout);

    Ainvm = model.A\model.m2; %Ux1
    %Ainvm = model.Ainv*model.m2;
    C = Ainvm*Ainvm' + model.Ainv;
    Kuuinv = blkdiag(model.Kuuinv{:});
    for d = 1:model.nout,
        indd = d*ones(1,length(model.m{d}));
        EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2) + diag(model.etadq(d,:));
        EZ22 = EZ2(model.indXu,model.indXu);
        EZ = model.etadq(indd,model.indXu);
        
        Kfdu = cell2mat( model.Kfu(d,:) );

        dKfu{d} = (EZ' .* (Ainvm * (model.m{d}'*model.beta(d)))... 
            - (EZ22 .* (C - Kuuinv))*Kfdu'*model.beta(d))'; %TODO: transpose
        
        dKff(d,:) = cellfun( @(x) -.5*x*eye(length(model.m{d})), num2cell(model.etadq(d,:)*model.beta(d)), 'UniformOutput', false );
        
        dKSigma(d) = trace(model.m{d}*Ainvm'*(EZ.*Kfdu)') + .5*( -trace(Kfdu*( EZ22.*C )*Kfdu')...
            - sum(model.m{d}.^2)... 
            - sum(sum(repmat(model.etadq(d,:),length(model.m{d}),1).*cell2mat( model.Kff(d,:))))... 
            + trace( Kfdu*(EZ22.*Kuuinv)*Kfdu' ) + model.sizeX(d)/model.beta(d) );
    end
    
    dKfu = mat2cell(cell2mat(dKfu), cellfun('length',model.m), cellfun(@(x) size(x,1), model.latX));
    %PKuuinv = jitChol(model.P*Kuuinv);
    %dKuu = .5*(-C + Kuuinv - PKuuinv*PKuuinv');
    dKuu = .5*(-C + Kuuinv - Kuuinv*model.P*Kuuinv);
    dKuu = mat2cell(dKuu, cellfun(@(x) size(x,1), model.latX), cellfun(@(x) size(x,1), model.latX));
    induu = 1:model.nlf+1:model.nlf*model.nlf;
    dKuu = dKuu(induu) ; %Extract only the block diagonal terms

%     %% As Mauricio did
%     Ainv = cell2mat(model.Ainv);
%     AinvKuyDinvy = cell2mat(model.AinvKuyDinvy);
%     Kyu = cell2mat(model.Kfu);
%     C = mat2cell(Ainv + (AinvKuyDinvy*AinvKuyDinvy'), model.k*ones(1,model.nlf), model.k*ones(1,model.nlf));
%     CKuy = mat2cell(cell2mat(C)*Kyu', model.k*ones(1,model.nlf), cellfun('length',model.m));
%     
%     dKff = cell(model.nout,model.nlf);
%     %if isfield(model, 'useKernDiagGradient') && model.useKernDiagGradient
%     for k=1:model.nout,
%         for r = 1:model.nlf,
%             dKff{k,r} =  -0.5*diag(model.Dinv{k});
%         end
%     end
%     %else
%     %    for k=1:model.nout,
%     %        dKff{k} =  -0.5*sparseDiag(model.Dinv{k});
%     %    end
%     %end
%     dKfu = cell(model.nout,model.nlf);
%     dKSigma = zeros(1,model.nout);
%     
%     for r =1:model.nlf,
%         for k= 1:model.nout,
%             dKfu{k,r} = (model.KuuinvKuy{r,k}*model.beta(k) - CKuy{r,k}*model.beta(k) + ...
%                 model.AinvKuyDinvy{r}*model.m{k}'*model.beta(k))';
%         end
%     end
%     for k =1:model.nout,
%         H = zeros(size(model.outX{k},1),1);
%         for r =1:model.nlf,
%             temp =  model.Kfu{k,r}*model.AinvKuyDinvy{r};
%             H = H + 2*temp.*model.m{k} - sum(model.Kfu{k,r}.*CKuy{r,k}',2);
%         end
%         dKSigma(k) = -0.5*(sum(model.Ktilde{k}) - (size(model.outX{k},1)*1/model.beta(k)...
%             - model.m{k}'*model.m{k} + sum(H)));
%     end
%     
%     KuuinvKuyQ = cell(model.nlf,model.nout);
%     for r =1:model.nlf,
%         for k= 1:model.nout,
%             tempoD = model.Dinv{k}(:,ones(1,model.k));
%             KuuinvKuyQ{r,k} = model.KuuinvKuy{r,k}.*tempoD';
%         end
%     end
%     dKuu = cell(model.nlf,1);
%     for r =1:model.nlf,
%         KuuinvKuyQKyuKuuinv = zeros(model.k);
%         dKuu{r} = zeros(model.k);
%         for k=1: model.nout,
%             KuuinvKuyQKyuKuuinv =  KuuinvKuyQKyuKuuinv ...
%                 + KuuinvKuyQ{r,k}*model.KuuinvKuy{r,k}';
%         end
%         dKuu{r} = 0.5*((model.Kuuinv{r} - C{r,r}) - KuuinvKuyQKyuKuuinv);
%     end
    
end