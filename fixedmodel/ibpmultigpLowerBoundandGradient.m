function [f,g] = ibpmultigpLowerBoundandGradient(model)

%Calculate Kuu and its inverse
for q = 1:model.nlf,
    fhandle = str2func([kern.type 'KuuCompute']);
    model.Kuu{q} = fhandle(model.kern, model.latX{q},q) + gamma(q)*eye(length(model.M));
    L = jitChol(model.Kuu);
    model.Kuu{q} = L'*L;
    model.logDetKuu(q) = 2.*sum(log(diag(L)));
    L = L\eye(size(L));
    model.Kuuinv{q} = L*L';
end

%Compute the psi statistics
model.cdq = zeros(model.nout, model.nlf);
model.Psi2 = cell(model.nout, model.nlf);
model.Psi1 = cell(model.nout, model.nlf);
model.P = 0.;
model.m2 = cell(model.nlf,1);

%% First phase
%Gather Psi statistics, comp. P and m2 and LowerBound
sKffq = zeros(model.nout, model.nlf);
for d = 1:model.nout,
    model.Psi2{d} = 0.;
    n = 1;
    EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2)...
        + diag(model.etadq(d,:));
    flag_end = false;
    while n < model.sizeX(d),
        nend = n + model.BatchSize;
        if nend > model.sizeX(d),
            nend = model.sizeX(d);
            flag_end = true;
        end
        indn = n:nend;
        X = model.outX{d}(indn,:);
        m = model.m{d}(indn);
        
        Kfut = zeros(length(indn),length(model.indXu)); %n:nendxQ.M
        for q = 1:model.nlf,
            if d == 1,
                model.m2{q} = 0.;
            end
            
            if ismepty(model.Psi2{d,q}),
                model.Psi2{d,q} = 0.;
                model.Psi1{d,q} = 0.;
                model.cdq{d,q} = 0.;
            end
            
            %Compute Kff(q) and Kfu
            fhandle = str2func([kern.type 'KfuCompute']);
            Kfu = fhandle(model.kern, X, model.latX{q}, d, q);
            fhandle = str2func([kern.type 'KffCompute']);
            Kffq = fhandle(model.kern, X, d, q);
            
            model.Psi1{d,q} = model.Psi1{d,q} + Kfu'*m; %Mx1
            Psi2dq = Psi2dq + Kfu'*Kfu; %MxM
            sKffq(d,q) = sum(Kffq);
            Kfut(:,model.indXu == q) = Kfu;
            if flag_end,
                model.cdq{d,q} = model.beta( sKffq(d,q) - sum(sum( model.Kuuinv{q}.*Psi2dq )));
                model.m2{q} = model.m2{q} + (model.beta(d)*model.etadq(d,q))*model.Psi1{d,q}; %Mx1
            end
        end
        model.Psi2{d} = model.Psi2{d} + Kfut'*Kfut; %Q.MxQ.M
        n = nend;
    end
    model.P = model.P + model.beta(d)*(EZ2(model.indXu,model.indXu).*model.Psi2{d}); %Q.MxQ.M
end

model.A = blkdiag(model.Kuu{:}) + model.P;
[La, jitter] = jitChol(model.A);
if jitter > model.minjit,
    error('Jitter higher than one.')
end
model.A = La.'*La;
model.logDetA = 2.*sum(log(diag(La)));
Lainv = La\eye(size(La));
model.Ainv = Lainv*Lainv.';

Lainvm2 = Lainv'*model.m2;
% LowerBound computation
f = -.5*(sum(model.beta.*model.m2d) + Lainvm2'*Lainvm2 - model.logDetA + sum(model.logDetKuu)... 
    - sum(sum(model.cdq.*model.etadq)) );

%% Phase 2 - Compute the gradients
Ainvm = model.Ainv*model.m2; %Q.Mx1
C = Ainvm*Ainvm.' + model.Ainv;
Kuuinv = blkdiag(model.Kuuinv{:});
CmKuuinv = C - Kuuinv;

gKern = zeros(1,model.kern.nParams);
gBeta = zeros(1,model.nout);
for d = 1:model.nout,
    indd = d*ones(1,model.sizeX(d));
    EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2) + diag(model.etadq(d,:));
    EZ22 = EZ2(model.indXu,model.indXu);
    EZ = model.etadq(indd,model.indXu);
    
    C1 = model.beta(d)*(EZ'.*(Ainvm*m'));
    C2p = EZ22.*CmKuuinv;
    C2 = model.beta(d)*C2p;
    
    n = 1;
    while n < model.sizeX(d),
        nend = n + model.BatchSize;
        if nend > model.sizeX(d),
            nend = model.sizeX(d);
        end
        indn = n:nend;
        X = model.outX{d}(indn,:);
        m = model.m{d}(indn);
        
        for q = 1:model.nlf,
            %Compute Kfu
            fhandle = str2func([kern.type 'KfuCompute']);
            Kfu = fhandle(model.kern,X,model.latX{q},d,q);
            
            dKfu = (C1 - C2*Kfu.').';
            
            dKff = -.5*model.etadq(d,q)*model.beta(d)*eye(length(m));
            
            fhandle = str2func([kern.type 'KffKfuGradient']);
            gKern = gKern + fhandle(model.kern,X,model.latX{q},dKff,dKfu,d,q);
        end
    end
    
    %Beta gradients
    Kfduqm = cell2mat(model.Psi1(d,:)');
    gBeta(d) = sum(sum(Ainvm.* sum(model.etadq(d,model.indXu)'.*Kfduqm,2) ))...
        +.5*( -sum(sum(C2p.*model.Psi2{d}))...
        - model.md2(d)+ model.sizeX(d)/model.beta(d) - sum(model.etadq(d,:)*sKffq(d,:)));
end

factors = kernFactors(model.kern, 'gradfact');
gKern(factors.index) = gKern(factors.index).*factors.val;

for q = 1:model.nlf,
    dKuu = -.5*(CmKuuinv(indq,indq) + model.Kuuinv{q}*model.P(indq,indq)*model.Kuuinv{q});
    fhandle = str2func([kern.type 'KuuGradient']);
    gKern = gKern + fhandle(model.kern, model.latX{q}, dKuu, q);
end

fhandle = str2func([model.betaTransform 'Transform']);
gBeta = gBeta.*fhandle(model.beta, 'gradfact');

g = [gKern,gGamma,gBeta];
