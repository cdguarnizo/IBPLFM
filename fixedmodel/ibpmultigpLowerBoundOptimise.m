function f = ibpmultigpLowerBoundOptimise(model)

% IBPMULTIGPLOWERBOUND FOR IBP Parameters

% IBPMULTIGP
f = 0;

if model.isVarS,
    if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
        EZS2 = model.etadq.*(model.varSdq + model.muSdq.^2);
        EZS = model.etadq.*model.muSdq;
    else
        EZS2 = model.varSdq + model.muSdq.^2;
        EZS = model.muSdq;
    end
else
    if strcmp(model.sparsePriorType,'ard'), %Initkern on ard prior
       EZS2 = ones(size(model.muSdq));
       EZS = EZS2;
    else
        EZS2 = model.etadq;
        EZS = model.etadq;
    end
end

%% Lowerbound terms realted to u and data
if model.isVarU,
    for q = 1:model.nlf,
        mq = 0;
        Pqq = 0;
        for d = 1:model.nout,
            mq = mq + EZS(d,q)*model.beta(d)*(model.Kfu{d,q}.'*model.m{d});
            Pqq = Pqq + EZS2(d,q)*model.beta(d)*(model.Kfu{d,q}.'*model.Kfu{d,q});
        end
        
        % Add trace \sum mq E[uq]
        %mq = Kuuinv*mq
        mq = model.Kuu{q}\mq;
        f = f + trace(mq*model.Euast{q}.');
        
        Pqq = (model.Kuu{q}\Pqq)*model.Kuuinv{q};
        Euuast = model.Kuuast{q} + model.Euast{q}*model.Euast{q}.';
        
        % Add \sum Kuquqinv Euuast
        %f = f - 0.5*trace(model.Kuuinv{q}*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}.'));
        f = f - 0.5*trace(model.Kuu{q}\Euuast);

        % Add trace \sum \sum Pqq E[uq uq]
        f = f - 0.5*trace(Pqq*Euuast);
        k = 1:model.nlf;
        k(q) = [];
        for qp = k,
            Pqqp = 0;
            for d = 1:model.nout,
                Pqqp = Pqqp + (EZS(d,q)*EZS(d,qp)*model.beta(d))*(model.Kfu{d,q}.'*model.Kfu{d,qp});
            end
            Pqqp = model.Kuuinv{q}*Pqqp*model.Kuuinv{qp};
            % Add trace \sum \sum Pqq' E[uq' uq]
            f = f - 0.5*trace(Pqqp*(model.Euast{qp}*model.Euast{q}.'));
        end
        
        % Entropy H(u)
        f = f + 0.5*logdet(model.Kuuast{q}) + 0.5*model.k;
    end

    % Add logdet Kuu
    f = f - 0.5*sum(model.logDetKuu);
    % Add \sum \sum cdq
    f = f - 0.5*sum(sum( model.cdq.*EZS2 ));
    
else
    % Lower bound terms related to u
    f = f + .5*( (model.m2'*(model.A\model.m2)) - model.logDetA + sum(model.logDetKuu)... 
        - sum(sum( model.cdq.*model.etadq )) );
        
    % f = f + .5*( sum(model.sqrtAinvm.*model.sqrtAinvm) - model.logDetA + sum(model.logDetKuu)... 
    %    - sum(sum( model.cdq.*model.etadq )) );

%     %% As Mauricio did
%     f = -sum(model.sizeX)*log(2*pi);
%     f = f - model.logDetDT; % contribution of ln det D
%     f = f - model.traceDinvyy; % contribution of trace(inv D yy')
%     f = f + sum(model.logDetKuu); % contribution of ln det Kuu
%     f = f - model.logDetA; % contribution of ln det A
%     for k = 1:model.nlf, % contribution of trace(invD Kyu invA Kuy invD yy')
%         f = f + sum(model.sqrtAinvKuyDinvy{k}.*model.sqrtAinvKuyDinvy{k});
%     end
%     f = f - model.KtildeT; % Only meaningful if the approximation is DTCVAR
%     f = 0.5*f;

end

%% Lower Bound terms related to data
% Add 0.5 log Sigma_w
f = f + 0.5*sum(model.sizeX.*log(model.beta));
% Add trace (Sigma_w yy^{\top})
f = f - 0.5*sum(model.beta(model.indX)'.*cell2mat(model.m).^2);

if model.isVarS,
    %% Lowerbound terms related to spikes and slab prior
    if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
        % Add 0.5\sum \sum E[Zdq]
        f = f - 0.5*log(2*pi)*sum(sum(model.etadq));
        
        %  Add 0.5*\sum\sum E[Zdq][psi(adqast) - log(bdqast)]
        if model.gammaPrior
            f = f + 0.5*sum(sum(model.etadq.*(psi(model.adqast) - log(model.bdqast))));
        else
            f = f + 0.5*sum(log(model.gammadq(:)));
        end
    else
        % Add 0.5*D*Q*log(2*pi)
        f = f - 0.5*log(2*pi)*model.nout*model.nlf;
        
        %  Add 0.5*\sum\sum E[Zdq][psi(adqast) - log(bdqast)]
        if model.gammaPrior,
            f = f + 0.5*sum(sum(psi(model.adqast) - log(model.bdqast)));
        else
            f = f + 0.5*sum(log(model.gammadq(:) + (model.gammadq(:)==0.) ));
        end
    end
    %  Add 0.5 \sum\sum ((adqast/bdqast )E[ZdqS2dq]
    if model.gammaPrior,
        f = f - 0.5*sum(sum( ((model.adqast./model.bdqast)).*EZS2));
    else
        f = f - 0.5*sum(sum( model.gammadq));
    end
    
    %% Lowerbound terms related to Gamma distribution
    if model.gammaPrior,
        % Add \sum \sum log Gamma(adq)
        f = f - sum(sum(log(gamma(model.adq))));
        % Add \sum \sum adq log bdq
        f = f + sum(sum( model.adq.*log(model.bdq)));
        % Add \sum \sum (adq - 1)[psi(adqast) - log (bdqast)]
        f = f + sum(sum( (model.adq - 1).*(psi(model.adqast) -log(model.bdqast))));
        % Add bdq (adqast/bdqast)
        f = f - sum(sum(model.bdq.*(model.adqast./model.bdqast)));
    end
end

%% Lowerbound terms realted to IBP prior
if strcmp(model.sparsePriorType,'ibp')

    if model.IBPisInfinite,
        % E[p(z_dq)]
        % Add \sum \sum E[Zdq]E[log piq]
        f = f + sum(sum(model.etadq.*repmat(model.Elogpiq, model.nout, 1)));
        %  Add \sum \sum (1-E[Zdq])E[log(1-piq)]
        f = f + sum(sum((1 - model.etadq).*repmat(model.Elog1mProdpim, model.nout, 1)));
        % E[p(upsilon_q)]
        % Add (alpha - 1)\sum (psi(tau1) - psi(tau1 + tau2))
        f = f + (model.alpha - 1)*sum(psi(model.tau1) - psi(model.tau1 + model.tau2));
        % Add Q log(alpha)
        f = f + model.nlf*log(model.alpha);
    else
        % E[p(z_dq)]
        % Add \sum \sum E[Zdq]E[log piq]
        f = f + sum(sum(model.etadq.*repmat(model.Elogpiq, model.nout, 1)));
        %  Add \sum \sum (1-E[Zdq])E[log(1-piq)]
        f = f + sum(sum((1 - model.etadq).*repmat(model.Elog1mProdpim, model.nout, 1)));
        % Add -\sum psi(ta1 + tau2)
        f = f + sum(sum(repmat(psi(model.tau1 + model.tau2), model.nout, 1)));
        % E[p(pi_q)]
        % Add (alpha - 1)\sum (psi(tau1) - psi(tau1 + tau2))
        f = f + (model.alpha/model.nlf - 1)*sum(psi(model.tau1) - psi(model.tau1 + model.tau2));
        % Add Q log(alpha)
        f = f + model.nlf*log(model.alpha/model.nlf);
    end
    
elseif strcmp(model.sparsePriorType,'spikes')
    % Add \sum \sum E[Zdq][log pi]
    f = f + log(model.pi + (model.pi==0))*sum(model.etadq(:));
    %  Add \sum \sum (1-E[Zdq])[log(1-pi)]
    f = f + log(1 - model.pi + (model.pi==1))*sum(1 - model.etadq(:));
end

%% Entropy H(S,Z)
if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
    %Entropy H(Z) As Finale does
%         tmpeta = model.etadq + .5 * ( model.etadq == 0 ) - .5 * ( model.etadq == 1 );
%         tmp = -1 * tmpeta.* log( tmpeta ) - ( 1 - tmpeta ).* log( 1 - tmpeta );
%         tmp = tmp .* ( model.etadq > 0 ) .* ( model.etadq < 1 );
    
    %As Titsias does
    tmp = -model.etadq.*(log(model.etadq + (model.etadq==0)))...
        - (1-model.etadq).*log(1-model.etadq + (model.etadq==1));
    
    % Entropy H(S,Z)
    HSZ = sum(sum(tmp));
    
    if model.isVarS,
        HSZ = HSZ + sum(sum(0.5*model.etadq.*(log(2*pi*exp(1)*model.varSdq + (model.varSdq(:)==0.) ) )));
    end
    
    f = f + HSZ;
else
    %Entropy H(S) for ARD sparse prior
    f = f + 0.5*sum(log(2*pi*exp(1)*model.varSdq(:) + (model.varSdq(:)==0.)));
end


if strcmp(model.sparsePriorType,'ibp'),
    % Entroy H(upsilon) - Infinite or H(pi) for Finite
    Hupsi = sum( log((gamma(model.tau1).*gamma(model.tau2))./gamma(model.tau1+model.tau2)) ...
        - (model.tau1 - 1).*psi(model.tau1) - (model.tau2 - 1).*psi(model.tau2) ...
        + (model.tau1 + model.tau2 - 2).*psi(model.tau1+model.tau2));
    f = f + Hupsi;
end
