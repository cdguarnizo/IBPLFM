function model = ibpmultigpMomentsCompute(model)

% IBPMULTIGPMOMENTSCOMPUTE

% IBPMULTIGP

% Compute moments for multinomials qki
templateqki = zeros(model.nlf);
for q=1:model.nlf
   temp = psi(model.tau2(1:q)) - cumsum(psi(model.tau1(1:q) + model.tau2(1:q)));
   if q > 1
       temp = temp + [0 cumsum(psi(model.tau1(1:q-1)))];
   end
   templateqki(q, 1:q) = exp(temp)/sum(exp(temp));
end
for q = 1:model.nlf
    model.qki{q} = templateqki(q, 1:q);
    model.qki2{q} = templateqki(q:end, q);
end

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
        if i<model.nlf
           model.tau1(i) = model.tau1(i) + sum(sum((1 - model.etadq(:, i+1:end))...
               .*repmat(templateqki2(i+1:end, i+1)', model.nout, 1)));
        end
    model.tau2(i) = 1 + sum(sum((1 - model.etadq(:, i:end)).*repmat(model.qki2{i}', model.nout, 1)));
end

model = ibpmultigpComputeELog(model);

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
model.adqast = model.adq + 0.5*model.etadq;
model.bdqast = model.bdq + 0.5*model.etadq.*(model.varSdq + model.muSdq.^2);

%ffinal = ibpmultigpLowerBound(model);
%%%%%%%%%%%%%%%
% Here we test the update for adqast
%%%%%%%%%%%%%%%
% epsilon = 1e-6;
% adqast = model.adqast;
% model.adqast = adqast + epsilon;
% fp = ibpmultigpLowerBound(model);
% model.adqast = adqast - epsilon;
% fm = ibpmultigpLowerBound(model);
% delta = 0.5*(fp - fm)/epsilon;

%%%%%%%%%%%%%%%
% Here we test the update for bdqast
%%%%%%%%%%%%%%%
% model.adqast = adqast;
% epsilon = 1e-6;
% bdqast = model.bdqast;
% model.bdqast = bdqast + epsilon;
% fp = ibpmultigpLowerBound(model);
% model.bdqast = bdqast - epsilon;
% fm = ibpmultigpLowerBound(model);
% delta = 0.5*(fp - fm)/epsilon;

% Precomputation of terms 
for q=1:model.nlf
   model.KuuinvEuast{q, 1} = model.Kuuinv{q}*model.Euast{q};
   model.sqrtEuuast{q, 1} = jitChol(model.Euuast{q, q});
   model.sqrtKuuinvsqrtEuuast{q, 1} = model.sqrtKuuinv{q}.'*model.sqrtEuuast{q, 1}.';
end

varSdqInv = zeros(model.nout, model.nlf);
Egammadq =  zeros(model.nout, model.nlf);
Eloggammadq = zeros(model.nout, model.nlf);
varthetadq = zeros(model.nout, model.nlf);
pdq = zeros(model.nout, model.nlf);

varthetadq2 = varthetadq;
etadq2 = varthetadq;
varSdq2 = varthetadq;
delta = -100*ones(model.nout,model.nlf);
mod2 = model;


for d=1:model.nout
   for q=1:model.nlf
       Kfutildel = model.Kfutilde(d, :);
       Kfutildel(q) = [];
       KuuinvEuastl = model.KuuinvEuast;
       KuuinvEuastl(q) = [];
       KfutildelMat = cell2mat(Kfutildel);
       KuuinvEuastMat = cell2mat(KuuinvEuastl);
       model.yhatdq{d, q} = KfutildelMat*KuuinvEuastMat;
       model.KfuKuuinvEuast{d, q} = model.Kfu{d, q}*model.KuuinvEuast{q};
       model.KfuKuuinvsqrtEuuast{d, q} = model.KfusqrtKuuinv{d, q}*model.sqrtKuuinvsqrtEuuast{q};
       % Update variance Sdq given Zdq
       model.Lambdadq(d, q) = sum(sum(model.KfuKuuinvsqrtEuuast{d, q}.^2));       
       Egammadq(d,q) = model.adqast(d,q)/model.bdqast(d,q);
       varSdqInv(d,q) = model.beta(d)*model.Lambdadq(d,q) + Egammadq(d,q) ...
           + model.cdq(d,q);
       model.varSdq(d,q) = 1/varSdqInv(d,q);
       
       %Check evaluation of varSdq
        Pdqq = model.beta(d)*model.Kuuinv{q}*model.Kfu{d,q}.'*model.Kfu{d,q}*model.Kuuinv{q};
        varSdq2(d,q) = 1/(trace(Pdqq*(model.Kuuast{q}+model.Euast{q}*model.Euast{q}.'))...
            + model.adqast(d,q)/model.bdqast(d,q) + model.cdq(d,q));
       
       % Update mean Sdq given Zdq
       model.Psidq(d, q) = sum(model.KfuKuuinvEuast{d, q}.*model.yhatdq{d, q});
       model.yalphatildedq(d, q) = sum(model.KfuKuuinvEuast{d, q}.*model.y{d});
       pdq(d,q) = model.beta(d)*(model.yalphatildedq(d, q) - model.Psidq(d, q));
       
       %check pdq
       k = 1:model.nlf;
       k(q) = [];
       T2 = 0;
       for q1 = k
           T2 = T2 + mod2.etadq(d,q1)*mod2.muSdq(d,q1)*mod2.Kfu{d,q1}*mod2.Kuuinv{q1}*mod2.Euast{q1};
       end
       pdq2(d,q) = mod2.beta(d)*trace(mod2.Kuuinv{q}*mod2.Kfu{d,q}.'*(mod2.y{d} - T2)*mod2.Euast{q}.');

       model.muSdq(d,q) = model.varSdq(d,q)*pdq(d,q);
       % Update eta dq
       Eloggammadq(d,q) = psi(model.adqast(d, q)) - log(model.bdqast(d,q));
       varthetadq(d,q) = 0.5*log(model.varSdq(d,q)) ...
           + 0.5*varSdqInv(d,q)*(model.muSdq(d,q)^2) ...
           + model.Elogpiq(q) + 0.5*Eloggammadq(d,q) - model.Elog1mProdpim(q);
       
       model.etadq(d,q) = 1/(1 + exp(-varthetadq(d,q)));
       
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
%        epsilon = 1e-6;
%        temp = mod2;
%        etadq = model.etadq(d, q);
%        mod2.varSdq(d,q) = model.varSdq(d,q);
%        mod2.muSdq(d,q) = model.muSdq(d,q);
%        mod2.etadq(d, q) = etadq + epsilon;
%        mod2 = ibpmultigpUpdateA2(mod2);
%        fp = ibpmultigpLowerBound(mod2);
%        mod2.etadq(d, q) = etadq - epsilon;
%        mod2 = ibpmultigpUpdateA2(mod2);
%        fm = ibpmultigpLowerBound(mod2);
%        delta(d,q) = 0.5*(fp - fm)/epsilon;
%        mod2 = temp;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %Check Update of mudq
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       epsilon = 1e-6;
       temp = mod2;
       muSdq = model.muSdq(d, q);
       %mod2.varSdq(d,q) = model.varSdq(d,q); 
       mod2.muSdq(d, q) = muSdq + epsilon;
       %mod2 = ibpmultigpUpdateA2(mod2);
       fp = ibpmultigpLowerBound(mod2);
       mod2.muSdq(d, q) = muSdq - epsilon;
       %mod2 = ibpmultigpUpdateA2(mod2);
       fm = ibpmultigpLowerBound(mod2);
       delta(d,q) = 0.5*(fp - fm)/epsilon;
       mod2 = temp;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %Check Update of varSdq
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        epsilon = 1e-6;
%        varSdq = model.varSdq(d, q);
%        temp = mod2;
%        
%        mod2.varSdq(d, q) = varSdq + epsilon;
%        %mod2.ES2dq(d, q) = mod2.varSdq(d, q) + mod2.muSdq(d, q)^2;
%        %mod2.EZdqS2dq(d, q) = mod2.etadq(d,q)*(mod2.varSdq(d, q) + mod2.muSdq(d,q)^2);
%        %mod2 = ibpmultigpUpdateA2(mod2);
%        fp = ibpmultigpLowerBound2(mod2);
% 
%        mod2.varSdq(d, q) = varSdq - epsilon;
%        %mod2.ES2dq(d, q) = mod2.varSdq(d, q) + mod2.muSdq(d, q)^2;
%        %mod2.EZdqS2dq(d, q) = mod2.etadq(d,q)*(mod2.varSdq(d, q) + mod2.muSdq(d,q)^2);
%        %mod2 = ibpmultigpUpdateA2(mod2);
%        fm = ibpmultigpLowerBound2(mod2);
%        
%        delta(d,q) = 0.5*(fp - fm)/epsilon;
%        mod2 = temp;
   end    
end

delta = zeros(model.nlf,1);
mod2 = model;

% % Compute moments for q(uq)
% for d=1:model.nout
%     for q=1:model.nlf
%         model.Kfutilde{d, q} = model.EZdqSdq(d,q)*model.Kfu{d, q};
%     end
% end
% for q1 = 1:model.nlf
%     KuftSigmaKuft = zeros(model.k);
%     for d = 1:model.nout
%         const = model.beta(d)*model.EZdqS2dq(d, q1);
%         KufKfu = model.Kfu{d, q1}.'*model.Kfu{d, q1};
%         KuftSigmaKuft = KuftSigmaKuft + const*KufKfu;
%     end
%     if isfield(model, 'gamma') && ~isempty(model.gamma)
%         model.A{q1} = model.KuuGamma{q1} + KuftSigmaKuft;
%     else
%         model.A{q1} = model.Kuu{q1} + KuftSigmaKuft;
%     end
%     [model.Ainv{q1}, ~, ~, model.sqrtAinv{q1}] = pdinv(model.A{q1});
% end
% for q=1:model.nlf
%    KuqfSy = zeros(model.k, 1); 
%    for d=1:model.nout
%        KuqfSy = KuqfSy + model.beta(d)*(model.Kfutilde{d,q}.'*...
%            (model.y{d} - model.yhatdq{d,q}));
%    end
%    if isfield(model, 'gamma') && ~isempty(model.gamma)
%        model.Euast{q} = model.KuuGamma{q}*(model.Ainv{q}*KuqfSy);
%        KuusqrtAinv = model.KuuGamma{q}*model.sqrtAinv{q};       
%    else
%        model.Euast{q} = model.Kuu{q}*(model.Ainv{q}*KuqfSy);
%        KuusqrtAinv = model.Kuu{q}*model.sqrtAinv{q};
%    end
%    
%    model.Kuuast{q} = KuusqrtAinv*KuusqrtAinv';
%    
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    epsilon = 1e-6;
%    Euast = model.Euast{q};
%    temp = mod2;
%    mod2.Euast{q} = Euast + epsilon;
%    mod2.Kuuast{q} = model.Kuuast{q};
%    mod2 = ibpmultigpUpdateA2(mod2);
%    fp = ibpmultigpLowerBound(mod2);
%    mod2.Euast{q} = Euast - epsilon;
%    mod2 = ibpmultigpUpdateA2(mod2);
%    fm = ibpmultigpLowerBound(mod2);
%    delta(q) = 0.5*(fp - fm)/epsilon;
%    mod2 = temp;
% end
% delta'
% for q1=1:model.nlf
%     model.Euuast{q1, q1} = model.Kuuast{q1} + model.Euast{q1}*model.Euast{q1}.';
%     for q2=1:q1-1
%         model.Euuast{q1, q2} = model.Euast{q1}*model.Euast{q2}.';
%         model.Euuast{q2, q1} = model.Euuast{q1, q2}.';
%     end
% end

%Another version to calculate Euast u Kuuast
for d=1:model.nout
    for q=1:model.nlf
        %model.Kfutilde{d, q} = (model.muSdq(d,q)*model.etadq(d,q))*model.Kfu{d, q};
        model.Kfutilde{d, q} = model.EZdqSdq(d,q)*model.Kfu{d, q};
    end
end
for q1 = 1:model.nlf
    KuftSigmaKuft = zeros(model.k);
    for d = 1:model.nout
        %const = model.beta(d)*(model.muSdq(d,q1)^2 + model.varSdq(d,q1))*model.etadq(d,q1);
        const = model.beta(d)*model.EZdqS2dq(d, q1);
        KufKfu = model.Kfu{d, q1}.'*model.Kfu{d, q1};
        KuftSigmaKuft = KuftSigmaKuft + const*KufKfu;
    end
end

%Update  yhat and companions
for d=1:model.nout
   for q=1:model.nlf
       Kfutildel = model.Kfutilde(d, :);
       Kfutildel(q) = [];
       KuuinvEuastl = model.KuuinvEuast;
       KuuinvEuastl(q) = [];
       KfutildelMat = cell2mat(Kfutildel);
       KuuinvEuastMat = cell2mat(KuuinvEuastl);
       model.yhatdq{d, q} = KfutildelMat*KuuinvEuastMat;
       model.KfuKuuinvEuast{d, q} = model.Kfu{d, q}*model.KuuinvEuast{q};
       model.KfuKuuinvsqrtEuuast{d, q} = model.KfusqrtKuuinv{d, q}*model.sqrtKuuinvsqrtEuuast{q};
   end
end

model = ibpmultigpUpdateA(model);
mod2 = model;
delta = ones(model.nlf,1);
grad = delta;
ana = delta;

for q=1:model.nlf
    model.Kuuast{q} = zeros(size(model.P{q,q}));
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        Kuuinv = model.KuuGamma{q}\eye(size(model.P{q,q}));
        model.Kuuast{q} = (model.P{q,q} + Kuuinv)\eye(size(model.P{q,q}));
    else
        Kuuinv = model.Kuu{q}\eye(size(model.P{q,q}));
        model.Kuuast{q} = (model.P{q,q} + Kuuinv)\eye(size(model.P{q,q}));
    end
    
    KuqfSy = zeros(model.k, 1);
    for d=1:model.nout
        KuqfSy = KuqfSy + model.beta(d)*(model.Kfutilde{d,q}.'*(model.m{d} - model.yhatdq{d,q}));
    end
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        model.Euast{q} = model.Kuuast{q}*(model.KuuGamma{q}\KuqfSy);
    else
        model.Euast{q} = model.Kuuast{q}*(model.Kuu{q}\KuqfSy);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    epsilon = 1e-6;
    Euast = model.Euast{q};
    temp = mod2;
    mod2.Kuuast{q} = model.Kuuast{q};
    mod2.Euast{q} = Euast + epsilon;
    %mod2 = updateEuuast(mod2);
    fp = ibpmultigpLowerBound2(mod2);
    mod2.Euast{q} = Euast - epsilon;
    %mod2 = updateEuuast(mod2);
    fm = ibpmultigpLowerBound2(mod2);
    delta(q) = 0.5*(fp - fm)/epsilon;
    mod2 = temp;
    
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
%delta'

%calculation Euuast
model = updateEuuast(model);


%Update model
model = ibpmultigpUpdateA(model);


function mod = updateEuuast(mod)
for q1=1:mod.nlf
    mod.Euuast{q1, q1} = mod.Kuuast{q1} + mod.Euast{q1}*mod.Euast{q1}.';
    for q2=1:q1-1
        mod.Euuast{q1, q2} = mod.Euast{q1}*mod.Euast{q2}.';
        mod.Euuast{q2, q1} = mod.Euuast{q1, q2}.';
    end
end
