function model = ibpmultigpMomentsCompute(model)

% IBPMULTIGPMOMENTSCOMPUTE

% IBPMULTIGP

if model.debug,
    LB1 = ibpmultigpLowerBound(model);
end

if strcmp(model.sparsePriorType,'ibp'),
    if model.force_posUpdate,
        LBt = ibpmultigpLowerBound(model);
        tau1 = model.tau1;
        tau2 = model.tau2;
        Elog1mProdpim = model.Elog1mProdpim;
        Elogpiq = model.Elogpiq;
    end
    
    sum_eta = sum(model.etadq, 1);
    D_sum_eta = model.nout - sum_eta;
    
    if model.IBPisInfinite,
        %% Update of q_{ki} and tau
        %for k = 1:model.nlf,
        % Compute moments for multinomials qki
        templateqki = zeros(model.nlf);
        tau = [model.tau1; model.tau2];
        psi_tau = psi(tau);         % psi(t_k:)
        psi_sum = psi(sum(tau));    % psi(t_k1+tk2)
        psi_tau1_cs  = [0 cumsum(psi_tau(1, 1:model.nlf-1))];
        psi_cs = cumsum(psi_sum);
        temp = psi_tau(2,:) + psi_tau1_cs - psi_cs;
        temp = exp(temp - max(temp));
        for q = 1:model.nlf,
            templateqki(q, 1:q) = temp(1:q)/sum(temp(1:q));
        end
        
        % Compute moments for q(upsilon)
        for q = 1:model.nlf,
            model.tau1(q) = model.alpha + sum(sum_eta(q:model.nlf)) +...
                D_sum_eta(q+1:model.nlf)*sum(templateqki(q+1:end,q+1:end),2);
            model.tau2(q) = 1 + D_sum_eta(q:model.nlf)*templateqki(q:model.nlf,q);
        end
        
        % elseif strcmp(model.sparsePriorType,'spikes')
        %     %Update of pi for Spikes prior type
        %     model.pi = sum(model.etadq(:))/(model.nout*model.nlf);
        
        %     %%%%%%%%%%%%
        %     % Here we test update for pi
        %     %%%%%%%%%%%%
        %     epsilon = 1e-6;
        %     p = model.pi;
        %     model.pi = p + epsilon;
        %     fp = ibpmultigpLowerBound2(model);
        %     model.pi = p + epsilon;
        %     fm = ibpmultigpLowerBound2(model);
        %     delta = 0.5*(fp - fm)/epsilon;
        %     model.pi = p;
        
        %Update E[log(1-pi)] and E[log(pi)]
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
            qd = exp(tmp(1:q) - max(tmp(1:q)));
            qd = qd/sum(qd);
            
            model.Elog1mProdpim(q) = sum(qd.*(tmp(1:q)-log(qd)));
        end
    else % For finite version of IBP
        % Update for q(pi)
        model.tau1 = model.alpha/model.nlf + sum_eta;
        model.tau2 = 1 + D_sum_eta;
        
        model.Elogpiq = psi(model.tau1);
        model.Elog1mProdpim = psi(model.tau2);
    end
    
    if model.force_posUpdate,
        LB = ibpmultigpLowerBound(model);
        if (LB - LBt < 0.),
            model.tau1 = tau1;
            model.tau2 = tau2;
            model.Elogpiq = Elogpiq;
            model.Elog1mProdpim = Elog1mProdpim;
        else
            LBt = LB;
        end
    end
end

if model.debug,
    LB2 = ibpmultigpLowerBound(model);
    fprintf('After Updating q(pi),  Bound inc.: %f\n',LB2-LB1);
    LB1 = LB2;
end

% LB1 = ibpmultigpLowerBound(model) - LB1;
% if LB1<0,
%     model.tau1 = temp.tau1;
%     model.tau2 = temp.tau2;
%     model.Elog1mProdpim = temp.Elog1mProdpim;
%     model.Elopiq = temp.Elogpiq;
% end

% Fold = ibpmultigpLowerBound2(mod2);
% Fnew = ibpmultigpLowerBound2(model);
% fprintf('For tau1 and tau2: %f\n',Fnew-Fold)

% %%%%%%%%%%%%
% % Here we test update for the tau1 and tau2
% %%%%%%%%%%%%
% epsilon = 1e-6;
% qnum = 2;
% tau1 = model.tau1;
% %tau2 = model.tau2;
% model.tau1(qnum) = tau1(qnum) + epsilon;
% %model.tau2(qnum) = tau2(qnum) + epsilon;
% model = ibpmultigpComputeELog(model);
% fp = ibpmultigpLowerBound(model);
% model.tau1(qnum) = tau1(qnum) - epsilon;
% %model.tau2(qnum) = tau2(qnum) - epsilon;
% model = ibpmultigpComputeELog(model);
% fm = ibpmultigpLowerBound(model);
% delta = 0.5*(fp - fm)/epsilon;

%% Update of q(gamma)
% Compute moments for q(gamma)
if model.gammaPrior && model.isVarS,
    if strcmp(model.sparsePriorType,'spikes') || strcmp(model.type,'ibp')
        model.adqast = model.adq + 0.5*model.etadq;
        model.bdqast = model.bdq + 0.5*model.etadq.*(model.varSdq + model.muSdq.^2);
    else
        model.adqast = model.adq + 0.5;
        model.bdqast = model.bdq + 0.5*(model.varSdq + model.muSdq.^2);
    end
    
    if model.debug,
        LB2 = ibpmultigpLowerBound(model);
        fprintf('After Updating q(gamma), Bound inc.: %f\n',LB2-LB1);
        LB1 = LB2;
    end
end
% Fold = ibpmultigpLowerBound2(mod2);
% Fnew = ibpmultigpLowerBound2(model);
% fprintf('For adqast and bdqast: %f\n',Fnew-Fold)

%ffinal = ibpmultigpLowerBound(model);
%%%%%%%%%%%%%%%
% Here we test the update for adqast
%%%%%%%%%%%%%%%
% mod2=model;
% delta = zeros(model.nout, model.nlf);
% epsilon = 1e-6;
% for d=1:model.nout,
%     for q = 1:model.nlf,
%     adqast = model.adqast(d,q);
%     bdqast = model.bdqast(d,q);
%     temp = mod2;
%     mod2.bdqast(d,q) = bdqast;
%     mod2.adqast(d,q) = adqast + epsilon;
%     fp = ibpmultigpLowerBound2(mod2);
%     mod2.adqast(d,q) = adqast - epsilon;
%     fm = ibpmultigpLowerBound2(mod2);
%     delta(d,q) = 0.5*(fp - fm)/epsilon;
%     mod2 = temp;
%     end
% end

%%%%%%%%%%%%%%%
% Here we test the update for bdqast
%%%%%%%%%%%%%%%
% delta = zeros(model.nout, model.nlf);
% epsilon = 1e-6;
% for d=1:model.nout,
%     for q = 1:model.nlf,
%     adqast = model.adqast(d,q);
%     bdqast = model.bdqast(d,q);
%     temp = mod2;
%     mod2.adqast(d,q) = adqast;pdinv(Pqq + model.Kuuinv{q})
%     mod2.bdqast(d,q) = bdqast + epsilon;
%     fp = ibpmultigpLowerBound2(mod2);
%     mod2.bdqast(d,q) = bdqast - epsilon;
%     fm = ibpmultigpLowerBound2(mod2);
%     delta(d,q) = 0.5*(fp - fm)/epsilon;
%     mod2 = temp;
%     end
% end

if ~model.Opteta,
    %temp.etadq = model.etadq;
    
    %% Update of q(S,Z)
    %Eloggammadq = zeros(model.nout, model.nlf);
    pdq = zeros(model.nout, model.nlf);
    
    if model.isVarS,
        %muSdq = model.muSdq;
        %varSdq = model.varSdq;
        %varSdqInv = zeros(model.nout, model.nlf);
        if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
            etadq = model.etadq;
            varthetadq = zeros(model.nout, model.nlf);
        end
    else % IBP without prior on S
        varthetadq = zeros(model.nout, model.nlf);
        etadq = model.etadq;
    end
    
    if model.Trainvar,
        %Kuuinv Euuast Kuuinv
        temp = blkdiag(model.Kuuinv{:})*cell2mat(model.Euuast)*blkdiag(model.Kuuinv{:});
        temp = mat2cell(temp, model.sizeXu, model.sizeXu);
        for d=1:model.nout,
            for q=1:model.nlf,
                mdqEuast = model.beta(d)*(model.Euast{q}'*(model.Kuuinv{q}*model.Psi1{d,q}));
                
                % Calculate pdq
                indq = model.indXu == q;
                Psi2dq = model.Psi2{d}(indq,indq);
                pdq(d,q) = 0.;
                for q1 = [1:q-1,q+1:model.nlf],
                    indqp = model.indXu == q1;
                    Psi2dqqp = model.Psi2{d}(indq,indqp);
                    pdq(d,q) = pdq(d,q) + (etadq(d,q1)*model.beta(d))*...
                        sum(sum(Psi2dqqp'.*temp{q1,q}));
                end
                
                % Update etadq
                varthetadq(d,q) = mdqEuast -.5*(model.beta(d)*sum(sum(Psi2dq'.*temp{q,q})) + model.cdq(d,q))...
                    - pdq(d,q) + model.Elogpiq(q) - model.Elog1mProdpim(q);
                model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
            end
            
            if ~strcmp(model.sparsePriorType,'ard') && isnan(model.etadq(d,q)),
                error('etadq is NaN')
            end
            
            if model.force_posUpdate,
                LB = ibpmultigpLowerBound(model);
                if LB - LBt > 0,
                    LBt = LB;
                else
                    model.etadq(d,q) = etadq(d,q);
                end
            end
        end
    end
    
    if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp'),
        if any(isnan(model.etadq)),
            fprintf('NaN found in model.etadq\n')
        end
        model.etadq = real(model.etadq);
    end
    
    if isfield(model, 'muSdq'),
        sumS = sum(model.muSdq(:));
        if sumS == 0.,
            warning('Sensitivity values are zero.')
        end
    end
    
    if isfield(model, 'etadq'),
        sumS = sum(model.etadq(:));
        if sumS == 0.,
            warning('E[Z] is zero.')
        end
    end
    
    if model.debug,
        LB2 = ibpmultigpLowerBound(model);
        fprintf('After Updating q(S,Z), Bound inc.: %f\n',LB2-LB1);
        LB1 = LB2;
    end
    
    %% Update q(u)
    [model.Euast, model.Kuuast, model.logDetKuuast, model.Euuast] = ibpmultigpUpdateLatent(model,1);
    
    if model.debug,
        LB2 = ibpmultigpLowerBound(model);
        fprintf('After Updating q(u),   Bound inc.: %f\n\n', LB2-LB1);
    end
end