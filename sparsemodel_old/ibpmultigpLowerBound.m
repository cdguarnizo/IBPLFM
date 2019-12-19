function f = ibpmultigpLowerBound(model)

% IBPMULTIGPLOWERBOUND

% IBPMULTIGP
f = 0;
% Add trace \sum mq E[uq] 
for q = 1:model.nlf
    f = f + sum(model.mq{q}.*model.Euast{q});
end
% Add trace \sum \sum Pqq' E[uq'uq]
for q1 = 1:model.nlf
    f = f - 0.5*trace(model.P{q1, q1}*(model.Kuuast{q1} + model.Euast{q1}*model.Euast{q1}.'));
    k = 1:model.nlf;
    k(q1) = [];
    for q2 = k
        f = f - 0.5*trace(model.P{q1, q2}*(model.Euast{q2}*model.Euast{q1}.'));
        %f = f - 0.5*trace(model.P{q1, q2}*model.Euuast{q2,q1});
    end
end
% Add \sum Kuquqinv Euuast
for q = 1:model.nlf
   f = f - 0.5*trace(model.Kuuinv{q}*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}.')); 
   %f = f - 0.5*trace(model.Kuuinv{q}*model.Euuast{q,q}); 
end
% Add logdet Kuu
f = f - 0.5*sum(model.logDetKuu);
% Add (ND/2)log 2pi
f = f - 0.5*sum(model.sizeX)*log(2*pi);
for d = 1:model.nout
    % Add 0.5 log Sigma_w
    f = f + 0.5*model.sizeX(d)*log(model.beta(d)); 
    % Add trace (Sigma_w yy^{\top})
    f = f - 0.5*model.beta(d)*sum(model.m{d}.^2); %DIDIT change .m by .y
end
% Add 0.5\sum \sum E[Zdq]
%f = f - 0.5*log(2*pi)*sum(sum(model.etadq));
% Add \sum \sum E[Zdq]E[log piq]
f = f + sum(sum(model.etadq.*repmat(model.Elogpiq, model.nout, 1)));
%  Add 0.5*\sum\sum E[Zdq][psi(adqast) - log(bdqast)]
f = f + 0.5*sum(sum(model.etadq.*(psi(model.adqast) - log(model.bdqast))));
%  Add 0.5 \sum\sum ((adqast/bdqast + cdq)E[ZdqS2dq]
%f = f - 0.5*sum(sum(((model.adqast./model.bdqast) + model.cdq).*model.EZdqS2dq));
f = f - 0.5*sum(sum(((model.adqast./model.bdqast) + model.cdq).*...
    model.etadq.*(model.varSdq + model.muSdq.^2))); %DIDIT change EZdqS2dq

%  Add \sum \sum (1-E[Zdq])E[log(1-piq)]
f = f + sum(sum((1 - model.etadq).*repmat(model.Elog1mProdpim, model.nout, 1)));
% Add \sum \sum log Gamma(adq)
f = f - sum(sum(log(gamma(model.adq))));
% Add \sum \sum adq log bdq
f = f + sum(sum( model.adq.*log(model.bdq)));
% Add \sum \sum (adq - 1)[psi(adqast) - log (bdqast)]
f = f + sum(sum( (model.adq - 1).*(psi(model.adqast) -log(model.bdqast))));
% Add bdq (adqast/bdqast)
f = f - sum(sum(model.bdq.*(model.adqast./model.bdqast)));
% Add (alpha - 1)\sum (psi(tau1) - psi(tau1 + tau2))
f = f + (model.alpha - 1)*sum(psi(model.tau1) - psi(model.tau1 + model.tau2));
% Add Q log(alpha)
f = f + model.nlf*log(model.alpha);
% Add entropy H(u)
for q = 1:model.nlf
   f = f + 0.5*logdet(model.Kuuast{q}) + 0.5*model.k;
end

%Entropy H(Z) As Finale does
tmpeta = model.etadq + .5 * ( model.etadq == 0 ) - .5 * ( model.etadq == 1 );
tmp = -1 * tmpeta.* log( tmpeta ) - ( 1 - tmpeta ).* log( 1 - tmpeta );
tmp = tmp .* ( model.etadq > 0 ) .* ( model.etadq < 1 );

% Entropy H(S,Z)
HSZ = sum(sum(0.5*tmpeta.*(log(2*pi*exp(1)*model.varSdq)))) + sum(sum(tmp));

%HSZ = sum(sum(0.5*model.etadq.*(log(2*pi*exp(1)*model.varSdq)))) - ...
%     sum(sum(model.etadq.*(log(model.etadq)) + (1-model.etadq).*log(1-model.etadq)));

%  HSZ = sum(sum(model.etadq.*(0.5*log(model.varSdq)+0.5*(1+log(2*pi))))) - ...
%      sum(sum(model.etadq.*(log(model.etadq)) + (1-model.etadq).*log(1-model.etadq)));
%HSZ = sum(sum(model.etadq.*(0.5*log(model.varSdq)+0.5))) - ...
%    sum(sum(model.etadq.*(log(model.etadq)) + (1-model.etadq).*log(1-model.etadq)));
f = f + HSZ;
%%% Entroy H(upsilon)
Hupsi = sum( log((gamma(model.tau1).*gamma(model.tau2))./gamma(model.tau1+model.tau2)) ...
    - (model.tau1 - 1).*psi(model.tau1) - (model.tau2 - 1).*psi(model.tau2) ...
    + (model.tau1 + model.tau2 - 2).*psi(model.tau1+model.tau2));
f = f + Hupsi;
%%% Entropy H(gamma)
Hgamma = sum(sum(log(gamma(model.adqast)) - (model.adqast-1).*psi(model.adqast) ...
    - log(model.bdqast) + model.adqast));
f = f + Hgamma;
%%%

%f = -f;