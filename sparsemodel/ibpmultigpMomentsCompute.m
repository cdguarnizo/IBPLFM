function model = ibpmultigpMomentsCompute(model)

% IBPMULTIGPMOMENTSCOMPUTE

% IBPMULTIGP

if model.debug,
    LB1 = ibpmultigpLowerBound(model);
end


if strcmp(model.sparsePriorType,'ibp'),
    
    sum_eta = sum(model.etadq);
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
            %model.qki{q} = templateqki(q, 1:q);
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
end

if model.debug,
    LB2 = ibpmultigpLowerBound(model);
    fprintf('After Updating q(pi), Bound inc.: %f\n',LB2-LB1);
    LB1 = LB2;
end

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
        fprintf('After Updating q(gamma), Bound inc.: %f\n',LB2-LB1)
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
%% Update of q(S,Z)
Eloggammadq = zeros(model.nout, model.nlf);
pdq = zeros(model.nout, model.nlf);

if model.isVarS,
    muSdq = model.muSdq;
    varSdq = model.varSdq;
    varSdqInv = zeros(model.nout, model.nlf);
    if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
        etadq = model.etadq;
        varthetadq = zeros(model.nout, model.nlf);
    end
else % IBP without prior on S
    varthetadq = zeros(model.nout, model.nlf);
    etadq = model.etadq;
end

if model.Trainvar,
    for d=1:model.nout,
        for q=1:model.nlf,

            %Pdqq = model.beta(d)*model.Kuuinv{q}*model.Kfu{d,q}.'*model.Kfu{d,q}*model.Kuuinv{q};
            
            %Lambda = model.Kfu{d,q}*model.Kuuinv{q}*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}')*...
            %    (model.Kuuinv{q}*model.Kfu{d,q}');
            
            Lambda = model.KuuinvKuf{d,q}.'*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}')*...
                model.KuuinvKuf{d,q};
            
            % Update variance Sdq given Zdq
            %varSdqInv(d,q) = trace(Pdqq*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}'))...
            %    + model.adqast(d,q)/model.bdqast(d,q) + model.cdq(d,q);
            if model.isVarS,
                if model.gammaPrior,
                    varSdqInv(d,q) = model.beta(d)*trace(Lambda)...
                        + model.adqast(d,q)/model.bdqast(d,q) + model.cdq(d,q);
                    model.varSdq(d,q) = 1/varSdqInv(d,q);
                else
                    varSdqInv(d,q) = model.beta(d)*trace(Lambda)...
                        + model.gammadq(d,q) + model.cdq(d,q);
                    model.varSdq(d,q) = 1/varSdqInv(d,q);
                end
                
                if model.varSdq(d,q) < 0,
                    model.varSdq(d,q) = 0;
                    warning('Negative Variance')
                end
                
                % Update mean Sdq given Zdq
                % Calculate pdq
                k = 1:model.nlf;
                k(q) = [];
                T2 = 0;
                if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
                    for q1 = k, %yhat_d/q evaluation -> T2
                        %T2 = T2 + etadq(d,q1)*muSdq(d,q1)*model.Kfu{d,q1}*model.Kuuinv{q1}*model.Euast{q1};
                        T2 = T2 + etadq(d,q1)*muSdq(d,q1)*(model.KuuinvKuf{d,q1}.'*model.Euast{q1});
                    end
                else
                    for q1 = k, %yhat_d/q evaluation -> T2
                        %T2 = T2 + muSdq(d,q1)*model.Kfu{d,q1}*model.Kuuinv{q1}*model.Euast{q1};
                        T2 = T2 + muSdq(d,q1)*(model.KuuinvKuf{d,q1}.'*model.Euast{q1});
                    end
                end
                %pdq(d,q) = model.beta(d)*trace(KfuiKuu'*(model.m{d} - T2)*model.Euast{q}.');
                pdq(d,q) = model.beta(d)*trace(model.KuuinvKuf{d,q}*(model.m{d} - T2)*model.Euast{q}.');
                
                %Update muSdq
                model.muSdq(d,q) = model.varSdq(d,q)*pdq(d,q);
                
                % Update eta dq
                if model.gammaPrior,
                    Eloggammadq(d,q) = psi(model.adqast(d, q)) - log(model.bdqast(d,q));
                else
                    Eloggammadq(d,q) = log(model.gammadq(d,q));
                end
                
                if strcmp(model.sparsePriorType,'ibp'),
                    varthetadq(d,q) = muSdq(d,q)*pdq(d,q) - 0.5*(varSdq(d,q) + muSdq(d,q)^2)*...
                        varSdqInv(d,q) + 0.5*log(exp(1)*varSdq(d,q)) ...
                        + model.Elogpiq(q) + 0.5*Eloggammadq(d,q) - model.Elog1mProdpim(q);
                    model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
                elseif strcmp(model.sparsePriorType,'spikes'),
                    varthetadq(d,q) = muSdq(d,q)*pdq(d,q) - 0.5*(varSdq(d,q) + muSdq(d,q)^2)*...
                        varSdqInv(d,q) + 0.5*log(exp(1)*varSdq(d,q)) ...
                        + log(model.pi) + 0.5*Eloggammadq(d,q) - log(1-model.pi);
                    model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
                end
                
            else
                % Calculate pdq
                k = 1:model.nlf;
                k(q) = [];
                T2 = 0;
                for q1 = k, %yhat_d/q evaluation -> T2
                    %T2 = T2 + etadq(d,q1)*model.Kfu{d,q1}*model.Kuuinv{q1}*model.Euast{q1};
                    T2 = T2 + etadq(d,q1)*(model.KuuinvKuf{d,q1}.'*model.Euast{q1});
                end
                %pdq(d,q) = model.beta(d)*trace(KfuiKuu'*(model.m{d} - T2)*model.Euast{q}.');
                pdq(d,q) = model.beta(d)*trace(model.KuuinvKuf{d,q}*((model.m{d} - T2)*model.Euast{q}.'));
                
                % Update etadq
                varthetadq(d,q) = -.5*(model.beta(d)*trace(Lambda) + model.cdq(d,q))...
                    + pdq(d,q) + model.Elogpiq(q) - model.Elog1mProdpim(q);
                model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
            end

            if ~strcmp(model.sparsePriorType,'ard') && isnan(model.etadq(d,q)),
                error('etadq is NaN')
            end
            

            %        %Check varthetadq evaluation
            %        k= 1:model.nlf;
            %        k(q) =[];
            %        T=0;
            %        for q1 = k,
            %            T = T + trace(model.muSdq(d,q)*model.etadq(d,q1)*model.muSdq(d,q1)*model.Kuuinv{q}*model.Kfu{d,q}.'*model.beta(d)...
            %                *model.Kfu{d,q1}*model.Kuuinv{q1});
            %        end
            %        mdq = model.Kuuinv{q}*model.Kfu{d,q}.'*model.beta(d)*model.y{d};
            %        varthetadq2(d,q) = trace(model.muSdq(d,q)*mdq*model.Euast{q}.') - T -0.5*trace((model.varSdq(d,q)+model.muSdq(d,q)^2 ...
            %        )*(model.beta(d)*model.Kuuinv{q}*model.Kfu{d,q}'*model.Kfu{d,q}*model.Kuuinv{q})*(model.Kuuast{q}+model.Euast{q}*model.Euast{q}.')) ...
            %        -0.5*log(2*pi) + model.Elogpiq(q) + 0.5*Eloggammadq(d,q) - model.Elog1mProdpim(q) -0.5*(model.adqast(d,q)/model.bdqast(d,q)+model.cdq(d,q))...
            %        *(model.varSdq(d,q)+model.muSdq(d,q)^2) + 0.5*log(2*pi*exp(1)*mdoctoradoingenieria@utp.edu.coodel.varSdq(d,q));
            %        etadq2(d,q) = 1/(1 + exp(-varthetadq2(d,q)));
            
            %       fprintf('Iteracion %d, %d eta = %f vartheta = %f \n',d,q,model.etadq(d,q), varthetadq(d,q));
            %        if model.etadq(d,q) == 1,
            %            model.etadq(d,q) =0.999;
            %        end
            
            %Update other moments -- this is done in updateA2
            %model.ES2dq(d, q) = model.varSdq(d, q) + model.muSdq(d, q)^2;
            %model.EZdqSdq(d, q) = model.etadq(d, q)*model.muSdq(d, q);
            %model.EZdqS2dq(d, q) = model.etadq(d,q)*(model.varSdq(d, q) + model.muSdq(d,q)^2);
            %model.VZdqS2dq(d, q) = model.EZdqS2dq(d, q) - model.EZdqSdq(d, q)^2;
            %model.StdZdqS2dq(d, q) = sqrt(model.VZdqS2dq(d, q));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %        epsilon = 1e-6;
            %        EZdqSdq = model.EZdqSdq(d, q);
            %        model.EZdqSdq(d, q) = EZdqSdq + epsilon;
            %        model = ibpmultigpUpdateA(model);
            %        fp = ibpmultigpLowerBound(model);
            %        model.EZdqSdq(d, q) = EZdqSdq - epsilon;
            %        model = ibpmultigpUpdateA(model);
            %        fm = ibpmultigpLowerBound(model);
            %        delta(d,q) = 0.5*(fp - fm)/epsilon;
            %        model.EZdqSdq(d, q) = EZdqSdq;
            %        model = ibpmultigpUpdateA(model);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Check Update of eta
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                    epsilon = 1e-6;
            %                    temp = mod2;
            %                    %mod2.varSdq(d,q) = model.varSdq(d,q);
            %                    %mod2.muSdq(d,q) = model.muSdq(d,q);
            %                    mod2.etadq(d, q) = model.etadq(d, q) + epsilon;
            %                    %mod2 = ibpmultigpUpdateA2(mod2);
            %                    fp = ibpmultigpLowerBound2(mod2);
            %                    mod2.etadq(d, q) = model.etadq(d, q) - epsilon;
            %                    %mod2 = ibpmultigpUpdateA2(mod2);
            %                    fm = ibpmultigpLowerBound2(mod2);
            %                    delta(d,q) = 0.5*(fp - fm)/epsilon;
            %                    mod2 = temp;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Check Update of mudq
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                    epsilon = 1e-6;
            %                    temp = mod2;
            %                    mod2.muSdq(d, q) = model.muSdq(d,q) + epsilon;
            %                    fp = ibpmultigpLowerBound2(mod2);
            %                    mod2.muSdq(d, q) = model.muSdq(d,q) - epsilon;
            %                    fm = ibpmultigpLowerBound2(mod2);
            %                    delta(d,q) = 0.5*(fp - fm)/epsilon;
            %                    mod2 = temp;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Check Update of varSdq
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                    epsilon = 1e-6;
            %                    temp = mod2;
            %                    mod2.varSdq(d, q) = model.varSdq(d, q) + epsilon;
            %                    fp = ibpmultigpLowerBound2(mod2);
            %                    mod2.varSdq(d, q) = model.varSdq(d, q) - epsilon;
            %                    fm = ibpmultigpLowerBound2(mod2);
            %
            %                    delta(d,q) = 0.5*(fp - fm)/epsilon;
            %                    mod2 = temp;
        end
    end
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
    fprintf('After Updating q(S,Z), Bound inc.: %f\n',LB2-LB1)
    LB1 = LB2;
end

if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp'),
    if any(isnan(model.etadq)),
        fprintf('NaN found in model.etadq\n')
    end
    model.etadq = real(model.etadq);
end

%delta'

% %Check afte update LB increse
% Fold = ibpmultigpLowerBound2(mod2);
% Fnew = ibpmultigpLowerBound2(model);
% fprintf('For etadq, muSdq and varSdq: %f\n',Fnew-Fold)
% if Fnew<Fold,
%     %For muSdq
%     temp = mod2.muSdq;
%     mod2.muSdq = model.muSdq;
%     Fnew = ibpmultigpLowerBound2(mod2);
%     fprintf('\t For muSdq: %f\n',Fnew-Fold)
%     mod2.muSdq = temp;
%     %For muSdq
%     temp = mod2.varSdq;
%     mod2.varSdq = model.varSdq;
%     Fnew = ibpmultigpLowerBound2(mod2);
%     fprintf('\t For varSdq: %f\n',Fnew-Fold)
%     mod2.varSdq = temp;
%     %For muSdq
%     temp = mod2.etadq;
%     mod2.etadq = model.etadq;
%     Fnew = ibpmultigpLowerBound2(mod2);
%     fprintf('\t For etadq: %f\n',Fnew-Fold)
%     mod2.etadq = temp;
% end

%% Update of q(u)
%mod2 = model;
%delta = zeros(1,model.nlf);
Euast = model.Euast;
for q = 1:model.nlf,
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
        for d = 1:model.nout,
            Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.Kfu{d,q}.'*model.Kfu{d,q});
        end
    end
    %PqqKuuinv = Pqq*model.Kuuinv{q};
    %model.Kuuast{q} = (PqqKuuinv + eye(size(Pqq)))\model.Kuu{q};
    
    model.Kuuast{q} = model.Kuu{q}*((Pqq + model.Kuu{q})\model.Kuu{q});
    
    if any(isnan(model.Kuuast{q})),
        error('Nan in Kuuast')
    end
    %model.Kuuast{q} = model.Kuu{q}*((Pqq + model.Kuu{q})\model.Kuu{q});
    %model.Kuuast{q} = model.Kuu{q}*pdinv((Pqq + model.Kuu{q})*model.Kuu{q});
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Check gradident of Kuuast
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     epsilon = 1e-6;
    %     Kuuast = model.Kuuast{q};
    %     temp = mod2;
    %     mod2.Kuuast{q} = Kuuast + epsilon;
    %     fp = ibpmultigpLowerBound2(mod2);
    %     mod2.Kuuast{q} = Kuuast - epsilon;
    %     fm = ibpmultigpLowerBound2(mod2);
    %     delta(q) = 0.5*(fp - fm)/epsilon;
    %     mod2 = temp;
    
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
                        (model.KuuinvKuf{d,q2}'*Euast{q2});
                    %yhatdq = yhatdq + model.muSdq(d,q2)*...
                    %    (model.Kfu{d,q2}/model.Kuu{q2})*Euast{q2};
                end
                KuqfSy = KuqfSy + (model.beta(d)*model.muSdq(d,q))*(model.Kfu{d,q}.'...
                    *(model.m{d} - yhatdq));
            end
        else
            for q2 = k,
                %yhatdq = yhatdq + model.etadq(d,q2)*(model.Kfu{d,q2}*(model.Kuuinv{q2}*model.Euast{q2}));
                yhatdq = yhatdq + model.etadq(d,q2)*(model.KuuinvKuf{d,q2}.'*Euast{q2});
            end
            KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q))*(model.Kfu{d,q}'...
                *(model.m{d} - yhatdq));
        end
    end
    
    %model.Euast{q} = (PqqKuuinv + eye( size(Pqq) ))\KuqfSy;
    model.Euast{q} = model.Kuuast{q}*(model.Kuu{q}\KuqfSy);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Check gradident of Euast
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         epsilon = 1e-6;
%         Euast2 = model.Euast{q};
%         temp = mod2;
%         %mod2.Kuuast{q} = model.Kuuast{q};
%         mod2.Euast{q} = Euast2 + epsilon;
%         %mod2 = updateEuuast(mod2);
%         fp = ibpmultigpLowerBound(mod2);
%         mod2.Euast{q} = Euast2 - epsilon;
%         %mod2 = updateEuuast(mod2);
%         fm = ibpmultigpLowerBound(mod2);
%         delta(q) = 0.5*(fp - fm)/epsilon;
%         mod2 = temp;
    
%         %Check gradient by using Fechet derivative
%         mod2.Euast{q} = Euast{q};
%         %mod2 = updateEuuast(mod2);
%         fm = ibpmultigpLowerBound(mod2);
%         k=1:model.nlf;
%         k(q) = [];
%         T = 0;
%         for q1 = k,
%             T = T + mod2.P{q,q1}*mod2.Euast{q1};
%         end
%         grad(q) = sum(mod2.mq{q} - T - (mod2.P{q,q} + mod2.Kuuinv{q})*mod2.Euast{q});
%         ana(q) = fp-fm;
%         mod2.Euast{q} = temp.Euast{q};
end
end
if model.debug,
    LB2 = ibpmultigpLowerBound(model);
    fprintf('After Updating q(u), Bound inc.: %f\n\n',LB2-LB1)
end