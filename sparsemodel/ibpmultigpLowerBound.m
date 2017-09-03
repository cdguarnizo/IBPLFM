function f = ibpmultigpLowerBound(model)

% IBPMULTIGPLOWERBOUND

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
    if strcmp(model.sparsePriorType,'ard'),
        EZS2 = ones(size(model.muSdq));
        EZS = EZS2;
    else
        EZS2 = model.etadq;
        EZS = model.etadq;
    end
end

%% Lowerbound terms realted to u and data
if model.isVarU,
    temp = blkdiag(model.Kuuinv{:})*cell2mat(model.Euuast)*blkdiag(model.Kuuinv{:});
    temp = mat2cell(temp, model.sizeXu, model.sizeXu);
    for q = 1:model.nlf,
        indq = model.indXu==q;
        mq = 0;
        Pqq = 0;
        for d = 1:model.nout,
            mq = mq + (EZS(d,q)*model.beta(d))*model.Psi1{d,q};
            Pqq = Pqq + EZS2(d,q)*model.beta(d)*(model.Psi2{d}(indq,indq));
        end
        
        % Add trace \sum mq E[uq]
        mq = model.Kuuinv{q}*mq;
        f = f + mq.'*model.Euast{q};
        
        % Add \sum Kuquqinv Euuast
        f = f - 0.5*sum(sum(model.Kuuinv{q}'.*model.Euuast{q,q}));

        % Add trace \sum \sum Pqq E[uq uq]
        f = f - 0.5*sum(sum(Pqq.'.*temp{q,q}));
        k = 1:model.nlf;
        k(q) = [];
        for qp = k,
            indqp = model.indXu==qp;
            Pqqp = 0;
            for d = 1:model.nout,
                Pqqp = Pqqp + (EZS(d,q)*EZS(d,qp)*model.beta(d))*model.Psi2{d}(indq,indqp);
            end
            % Add trace \sum \sum Pqq' E[uq' uq]
            f = f - 0.5*sum(sum( Pqqp.'.*temp{qp,q} ));
        end
    end
    % Entropy H(u)
    % TODO: check this
    f = f + 0.5*(model.logDetKuuast + model.k*model.nlf);
    
    % Add logdet Kuu
    f = f - 0.5*sum(model.logDetKuu);
    % Add \sum \sum cdq
    f = f - 0.5*sum(sum( model.cdq.*EZS2 ));
else
    % Lower bound terms related to u
    Lainvm = model.Lainv'*model.m2;
    f = f + .5*( (Lainvm'*Lainvm) - model.logDetA + sum(model.logDetKuu)... 
       - sum( sum( model.cdq.*model.etadq )) );
end

%% Lower Bound terms related to data
% Add
f = f - 0.5*sum(model.sizeX)*log(2*pi);
% Add 0.5 log Sigma_w
f = f + 0.5*sum(model.sizeX.*log(model.beta));
% Add trace (Sigma_w yy^{\top})
if isfield(model,'UseMeanConstants') && model.UseMeanConstants,
    for d = 1:model.nout,
        f = f - 0.5*model.beta(d)*(sum((model.m{d}).^2) + model.sizeX(d)*model.mu(d).^2);
    end
else
    for d = 1:model.nout,
        f = f - 0.5*model.beta(d)*sum(model.m{d}.^2);
    end
end

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
        f = f - sum(sum(gammaln(model.adq)));
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
if model.Trainvar,
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
            HSZ = HSZ + sum(sum(0.5*model.etadq.*(log(2*pi*exp(1)*model.varSdq + (model.varSdq==0.) ) )));
        end
        
        f = f + HSZ;
    else
        %Entropy H(S) for ARD sparse prior
        f = f + 0.5*sum(log(2*pi*exp(1)*model.varSdq(:) + (model.varSdq(:)==0.)));
    end
end

if strcmp(model.sparsePriorType,'ibp'),
    % Entroy H(upsilon) - Infinite or H(pi) for Finite
    Hupsi = sum( gammaln(model.tau1) + gammaln(model.tau2) - gammaln(model.tau1+model.tau2) ...
        - (model.tau1 - 1).*psi(model.tau1) - (model.tau2 - 1).*psi(model.tau2) ...
        + (model.tau1 + model.tau2 - 2).*psi(model.tau1+model.tau2) );
    f = f + Hupsi;
end

%% Entropy H(gamma)
if model.gammaPrior && model.isVarS,
    Hgamma = sum(sum( gammaln(model.adqast) - (model.adqast-1).*psi(model.adqast) ...
        - log(model.bdqast) + model.adqast));
    f = f + Hgamma;
end

if ~isreal(f),
   warning('Imaginary part in lowerbound');
end

if isnan(f),
   warning('NaN in LB computation');
end