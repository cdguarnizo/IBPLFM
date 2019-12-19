function model = ibpmultigpMomentsCompute2(model)

% IBPMULTIGPMOMENTSCOMPUTE

% IBPMULTIGP

if strcmp(model.sparsePriorType,'ibp')
    if ~model.asFinale
        % Compute moments for multinomials qki
        templateqki = zeros(model.nlf);
        for q=1:model.nlf
            temp = psi(model.tau2(1:q)) - cumsum(psi(model.tau1(1:q) + model.tau2(1:q)));
            if q > 1
                temp = temp + [0 cumsum(psi(model.tau1(1:q-1)))];
                temp = temp - max(temp);
            end
            templateqki(q, 1:q) = exp(temp)/sum(exp(temp));
        end
        for q = 1:model.nlf
            model.qki{q} = templateqki(q, 1:q);
            model.qki2{q} = templateqki(q:end, q);
        end
        
        % Fold = ibpmultigpLowerBound2(mod2);
        % Fnew = ibpmultigpLowerBound2(model);
        % fprintf('For qki1 and qki2: %f\n',Fnew-Fold)
        
        %%%%%%%%%%%%
        % Here we test update for the multinomials
        %%%%%%%%%%%%
        % epsilon = 1e-6;
        % dens = 5;
        % qki = model.qki;
        % model.qki{dens} = qki{dens} + epsilon;
        % model.qki{dens} = model.qki{dens}/sum(model.qki{dens});
        % model = ibpmultigpComputeELog(model);
        % fp = ibpmultigpLowerBound(model);
        % model.qki{dens} = qki{dens} - epsilon;
        % model.qki{dens} = model.qki{dens}/sum(model.qki{dens});
        % model = ibpmultigpComputeELog(model);
        % fm = ibpmultigpLowerBound(model);
        % delta = 0.5*(fp - fm)/epsilon;
        %%%%%%%%%%%%
        
        % Compute moments for q(upsilon)
        templateqki2 = fliplr(cumsum(fliplr(templateqki), 2));
        for i=1:model.nlf
            model.tau1(i) = model.alpha + sum(sum(model.etadq(:, i:end)));
            if i < model.nlf
                model.tau1(i) = model.tau1(i) + sum(sum((1 - model.etadq(:, i+1:end))...
                    .*repmat(templateqki2(i+1:end, i+1)', model.nout, 1)));
            end
            model.tau2(i) = 1 + sum(sum((1 - model.etadq(:, i:end)).*repmat(model.qki2{i}', model.nout, 1)));
        end
    else
        % % As Finale does
        tau1_old = model.tau1;
        tau2_old = model.tau2;
        
        etadq = model.etadq;
        [N, K] = size( etadq );
        sum_n_nu = sum(etadq,1);
        N_minus_sum_n_nu = N - sum_n_nu;
        % Iterate through and update each tau(:,k)
        qs = zeros(K,K);
        for k = 1:K
            % First we compute q_k for k:K
            tau = [tau1_old;tau2_old];
            digamma_tau = psi( tau );
            digamma_sum = psi( sum( tau ) );
            digamma_tau1_cumsum = [ 0 cumsum( digamma_tau( 1 , 1:(K-1) ) ) ] ;
            digamma_sum_cumsum = cumsum( digamma_sum );
            exponent = digamma_tau( 2 , : ) + digamma_tau1_cumsum - digamma_sum_cumsum;
            unnormalized = exp(exponent - max(exponent));
            for m = k:K
                qs(m, 1:m) = unnormalized(1:m) / sum(unnormalized(1:m));
            end
            
            % Now that we have the q_k, update the tau(:,k)
            tau(1,k) = sum(sum_n_nu(k:K)) + N_minus_sum_n_nu(k+1:K) * ...
                sum(qs(k+1:end, k+1:end),2) + model.alpha;
            tau(2,k) = N_minus_sum_n_nu(k:K) * qs(k:K, k) + 1;
            
            % Finally commit the udpated tau and compute the lower bound if desired.
            %   model.tau(:,k) = tau(:,k);
            %   if params.compute_intermediate_lb
            %     lower_bounds_log(1,end+1) = ...
            %         compute_variational_lower_bound( params, X, alpha, sigma_a, sigma_n, model);
            %     lower_bounds_log(2,end) = 1;
            %     lower_bound_count = lower_bound_count + 1;
            %   end
        end
        for q = 1:model.nlf
            model.qki{q} = qs(q, 1:q);
            model.qki2{q} = qs(q:end, q);
        end
        model.tau1 = tau(1,:);
        model.tau2 = tau(2,:);
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

    %Update parameters related qki, tau1 and tau2
    model = ibpmultigpComputeELog(model);
    
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


% Compute moments for q(gamma)
if model.gammaPrior
    if strcmp(model.sparsePriorType,'spikes') || strcmp(model.type,'ibp')
        model.adqast = model.adq + 0.5*model.etadq;
        model.bdqast = model.bdq + 0.5*model.etadq.*(model.varSdq + model.muSdq.^2);
    else
        model.adqast = model.adq + 0.5;
        model.bdqast = model.bdq + 0.5*(model.varSdq + model.muSdq.^2);
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
%     mod2.adqast(d,q) = adqast;
%     mod2.bdqast(d,q) = bdqast + epsilon;
%     fp = ibpmultigpLowerBound2(mod2);
%     mod2.bdqast(d,q) = bdqast - epsilon;
%     fm = ibpmultigpLowerBound2(mod2);
%     delta(d,q) = 0.5*(fp - fm)/epsilon;
%     mod2 = temp;
%     end
% end

varSdqInv = zeros(model.nout, model.nlf);
Eloggammadq = zeros(model.nout, model.nlf);
varthetadq = zeros(model.nout, model.nlf);
pdq = zeros(model.nout, model.nlf);

%Save old values
muSdq = model.muSdq;
varSdq = model.varSdq;
if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'),
    etadq = model.etadq;
end
if model.Trainvar
    for d=1:model.nout
        for q=1:model.nlf
            %Pdqq = model.beta(d)*model.Kuuinv{q}*model.Kfu{d,q}.'*model.Kfu{d,q}*model.Kuuinv{q};
            Lambda = model.Kfu{d,q}*model.Kuuinv{q}*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}')*...
                (model.Kuuinv{q}*model.Kfu{d,q}');
            % Update variance Sdq given Zdq
            %varSdqInv(d,q) = trace(Pdqq*(model.Kuuast{q} + model.Euast{q}*model.Euast{q}'))...
            %    + model.adqast(d,q)/model.bdqast(d,q) + model.cdq(d,q);
            if model.gammaPrior
                varSdqInv(d,q) = model.beta(d)*trace(Lambda)...
                    + model.adqast(d,q)/model.bdqast(d,q) + model.cdq(d,q);
                model.varSdq(d,q) = 1/varSdqInv(d,q);
            else
                varSdqInv(d,q) = model.beta(d)*trace(Lambda)...
                    + model.gammadq(d,q) + model.cdq(d,q);
                model.varSdq(d,q) = 1/varSdqInv(d,q);
            end
            
            % Update mean Sdq given Zdq
            %Calculate pdq
            k = 1:model.nlf;
            k(q) = [];
            T2 = 0;
            if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
                for q1 = k %yhat_d/q evaluation -> T2
                    T2 = T2 + etadq(d,q1)*muSdq(d,q1)*model.Kfu{d,q1}*model.Kuuinv{q1}*model.Euast{q1};
                end
            else
                for q1 = k %yhat_d/q evaluation -> T2
                    T2 = T2 + muSdq(d,q1)*model.Kfu{d,q1}*model.Kuuinv{q1}*model.Euast{q1};
                end
            end
            pdq(d,q) = model.beta(d)*trace(model.Kuuinv{q}*model.Kfu{d,q}'*(model.m{d} - T2)*model.Euast{q}.');
            %Update muSdq
            model.muSdq(d,q) = model.varSdq(d,q)*pdq(d,q);
            
            % Update eta dq
            if model.gammaPrior
                Eloggammadq(d,q) = psi(model.adqast(d, q)) - log(model.bdqast(d,q));
            else
                Eloggammadq(d,q) = log(model.gammadq(d,q));
            end
            
            if strcmp(model.sparsePriorType,'ibp')
                varthetadq(d,q) = muSdq(d,q)*pdq(d,q) - 0.5*(varSdq(d,q) + muSdq(d,q)^2)*...
                    varSdqInv(d,q) + 0.5*log(exp(1)*varSdq(d,q)) ...
                    + model.Elogpiq(q) + 0.5*Eloggammadq(d,q) - model.Elog1mProdpim(q);
                model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
            elseif strcmp(model.sparsePriorType,'spikes')
                varthetadq(d,q) = muSdq(d,q)*pdq(d,q) - 0.5*(varSdq(d,q) + muSdq(d,q)^2)*...
                    varSdqInv(d,q) + 0.5*log(exp(1)*varSdq(d,q)) ...
                    + log(model.pi) + 0.5*Eloggammadq(d,q) - log(1-model.pi);
                model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
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
            %        *(model.varSdq(d,q)+model.muSdq(d,q)^2) + 0.5*log(2*pi*exp(1)*model.varSdq(d,q));
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
if any(isnan(model.etadq))
    fprintf('NaN found in model.etadq\n')
end
model.etadq = real(model.etadq);

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

Euast = model.Euast;

% if isfield(model, 'gamma') && ~isempty(model.gamma)
%     Kuu = model.KuuGamma;
% else
%     Kuu = model.Kuu;
% end
    
for q = 1:model.nlf
    %Update Kuuast
    model.Kuuast{q} = 0;
    Pqq = 0;
    if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
        for d=1:model.nout
            Pqq = Pqq + model.beta(d)*model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))...
                *(model.Kfu{d,q}.'*model.Kfu{d,q});
        end
    else
        for d=1:model.nout
            Pqq = Pqq + (model.beta(d)*(model.muSdq(d,q)^2+model.varSdq(d,q)))...
                *(model.Kfu{d,q}.'*model.Kfu{d,q});
        end
    end
    
    Pqq = model.Kuuinv{q}*Pqq*model.Kuuinv{q};
    model.Kuuast{q} = pdinv(Pqq + model.Kuuinv{q});
    
    %Pqq = Kuu{q}\Pqq;
    %model.Kuuast{q} = Kuu{q}*(Pqq + eye(size(Pqq))\eye(size(Pqq)));
    
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
    for d = 1:model.nout
        
        k = 1:model.nlf;
        k(q) = [];
        yhatdq = 0;
        if strcmp(model.sparsePriorType,'spikes') || strcmp(model.sparsePriorType,'ibp')
            for q2 = k
                yhatdq = yhatdq + (model.etadq(d,q2)*model.muSdq(d,q2))*...
                    (model.Kfu{d,q2}*(model.Kuuinv{q2}*Euast{q2}));
            end
            KuqfSy = KuqfSy + (model.beta(d)*model.etadq(d,q)*model.muSdq(d,q))*(model.Kfu{d,q}'...
                *(model.m{d} - yhatdq));
        else
            for q2 = k
                yhatdq = yhatdq + model.muSdq(d,q2)*...
                    (model.Kfu{d,q2}*(model.Kuuinv{q2}*Euast{q2}));
            end
            KuqfSy = KuqfSy + (model.beta(d)*model.muSdq(d,q))*(model.Kfu{d,q}.'...
                *(model.m{d} - yhatdq));
        end
    end
    %if isfield(model, 'gamma') && ~isempty(model.gamma)
    %    model.Euast{q} = model.Kuuast{q}*(model.KuuGamma{q}\KuqfSy);
    %else
    %     model.Euast{q} = model.Kuuast{q}*(model.Kuu{q}\KuqfSy);
    %end
    model.Euast{q} = model.Kuuast{q}*(model.Kuuinv{q}*KuqfSy);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Check gradident of Euast
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     epsilon = 1e-6;
    %     Euast2 = model.Euast{q};
    %     temp = mod2;
    %     %mod2.Kuuast{q} = model.Kuuast{q};
    %     mod2.Euast{q} = Euast2 + epsilon;
    %     %mod2 = updateEuuast(mod2);
    %     fp = ibpmultigpLowerBound2(mod2);
    %     mod2.Euast{q} = Euast2 - epsilon;
    %     %mod2 = updateEuuast(mod2);
    %     fm = ibpmultigpLowerBound2(mod2);
    %     delta(q) = 0.5*(fp - fm)/epsilon;
    %     mod2 = temp;
    
    %     %Check gradient by using Fechet derivative
    %     mod2.Euast{q} = Euast;
    %     %mod2 = updateEuuast(mod2);
    %     fm = ibpmultigpLowerBound(mod2);
    %     k=1:model.nlf;
    %     k(q) = [];
    %     T = 0;
    %     for q1 = k,
    %         T = T + mod2.P{q,q1}*mod2.Euast{q1};
    %     end
    %     grad(q) = sum(mod2.mq{q} - T - (mod2.P{q,q} + mod2.Kuuinv{q})*mod2.Euast{q});
    %     ana(q) = fp-fm;
    %     mod2.Euast{q} = temp.Euast{q};
end