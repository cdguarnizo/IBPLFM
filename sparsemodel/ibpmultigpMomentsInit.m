function model = ibpmultigpMomentsInit(model)

% IBPMULTIGPMOMENTSINIT

% IBPMULTIGP
if strcmp(model.sparsePriorType,'ibp')
    % Initialize tau
    model.tau1 = randi(3, 1, model.nlf);
    model.tau2 = randi(3, 1, model.nlf);

     %model.tau1 = model.alpha/model.nlf*ones(1,model.nlf) + ...
     %    .5*min(1, model.alpha)*(rand(1,model.nlf)-.5);
     %model.tau2 = ones(1,model.nlf) + .5*min(1, model.alpha)*(rand(1,model.nlf)-.5);

    
%     % Initialize qki
%     for q=1:model.nlf
%         model.qki{q} = rand(1, q);
%         model.qki{q} = model.qki{q}/sum(model.qki{q});
%     end
    % Compute expectations difficult terms IBP
    %model = ibpmultigpComputeELog(model);
    if model.IBPisInfinite,
        tau = [model.tau1; model.tau2];
        psi_tau = psi(tau);
        psi_sum = psi(sum(tau));
        cs_psi_tau1 = cumsum(psi_tau(1,:));
        cs_psi_tau10 = [ 0 cs_psi_tau1(:,1:end-1)];
        cs_psi_sum = cumsum(psi_sum);
        tmp = psi_tau(2,:) + cs_psi_tau10 - cs_psi_sum;
        for q = 1:model.nlf,
            % Compute E[log pi_q]
            model.Elogpiq(q) = cs_psi_tau1(q) - cs_psi_sum(q);
            % Compute E[log(1- prod(vm))]
            %qd = exp(tmp(1:q) - max(tmp(1:q)));
            qd = exp(tmp(1:q));
            qd = qd/sum(qd);
            
            model.Elog1mProdpim(q) = sum(qd.*(tmp(1:q)-log(qd)));
        end
    else
        model.Elogpiq = psi(model.tau1);
        model.Elog1mProdpim = psi(model.tau2);
    end
elseif strcmp(model.sparsePriorType,'spikes')
    model.pi = 0.5;
end

% Moments for q(uq)
for q=1:model.nlf,
   %model.Euast{q, 1} = randn(model.k, 1);
   %model.Kuuast{q, 1} = diag(ones(1,model.k));
   model.Euast{q, 1} = gsamp(zeros(model.k,1),model.Kuu{q},1)';
   model.Kuuast{q, 1} = model.Kuu{q};
end
% model.Euuast = cell(model.nlf);
% for q1=1:model.nlf      
%    model.Euuast{q1, q1} = eye(model.k) + model.Euast{q1}*model.Euast{q1}';
%    for q2 = 1:q1-1
%        model.Euuast{q1, q2} = model.Euast{q1}*model.Euast{q2}';
%        model.Euuast{q2, q1} = model.Euuast{q1, q2}'; 
%    end
% end

if model.gammaPrior,
    % Moments for q(gammadq) 
    model.adqast = randi(5, model.nout, model.nlf);
    model.bdqast = randi(5, model.nout, model.nlf);
end

if model.Trainvar && ~model.Opteta,
    if model.isVarS,
        if (strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')),
            % Initialize etadq (moment for q(Zdq))
            %model.etadq = zeros(model.nout, model.nlf) + 0.01;
            %model.etadq = 0.5*ones(model.nout, model.nlf) + 0.01*randn(model.nout, model.nlf);
            model.etadq = 0.9*ones(model.nout, model.nlf);
            %model.etadq(model.etadq < 0.2) = 0.2;
            %model.etadq(model.etadq > 0.8) = 0.8;
            
            %model.etadq = 0.9*ones(model.nout, model.nlf);
            %model.etadq = rand(model.nout, model.nlf);
        end
        % Initialize muSdq and varSdq (moments for q(Sdq|Zdq) )
        model.muSdq = ones(model.nout, model.nlf);
        model.varSdq = ones(model.nout, model.nlf);
    else
        model.etadq = ones(model.nout, model.nlf);
        %model.etadq = 0.5*ones(model.nout, model.nlf) + 0.1*randn(model.nout, model.nlf);
    end
end

if ~model.gammaPrior && model.isVarS,
    model.gammadq = model.muSdq.^2 + model.varSdq;
end