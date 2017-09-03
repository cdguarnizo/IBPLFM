function [uast, cK, logdet, Euuast] = ibpmultigpUpdateLatent(model, UpdateA)

Kuu = blkdiag(model.Kuu{:});

if UpdateA == 1,
    model.P = 0;
    model.m2 = cell(model.nlf,1);
    for d = 1:model.nout,
        EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2)...
            + diag(model.etadq(d,:));
        for q = 1:model.nlf,
            if d == 1,
                model.m2{q} = 0.;
            end
            model.m2{q} = model.m2{q} + (model.beta(d)*model.etadq(d,q))*model.Psi1{d,q}; %Mx1
        end
        model.P = model.P + model.beta(d)*(EZ2(model.indXu,model.indXu).*model.Psi2{d}); %Q.MxQ.M
    end
    model.m2 = cell2mat(model.m2); % Q.Mx1
    model.A = Kuu + model.P;
    [La, jitter] = jitChol(model.A);
    if jitter>model.minJit,
        error('Jitter higher than one.\n')
    end
    model.A = La.'*La;
    model.logDetA = 2.*sum(log(diag(La)));
    model.Lainv = La\eye(size(La));
    model.Ainv = model.Lainv*model.Lainv.';
end

temp = Kuu*model.Ainv;
Kuuast = temp*Kuu;
uast = temp*model.m2;

%Update Euuast
Euuast = Kuuast + uast*(uast');
Euuast = mat2cell(Euuast, model.sizeXu, model.sizeXu);

cK = mat2cell(Kuuast, model.sizeXu, model.sizeXu);
uast = mat2cell(uast, model.sizeXu);

Lu = jitChol(Kuuast);
logdet = 2.*sum(log(diag(Lu)));
