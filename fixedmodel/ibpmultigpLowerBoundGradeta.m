function geta = ibpmultigpLowerBoundGradeta(model)

% IBPMULTIGPLOWERBOUNDGRADETA

% IBPMULTIGP

%COPYRIGHT : Cristian Guarnizo, 2016


geta = zeros(1, model.nlf*model.nout);
getat = zeros(model.nout, model.nlf);

Ainvm = model.A\model.m2; %Ux1
%Ainvm = model.Ainv*model.m2;
Kuuinv = blkdiag(model.Kuuinv{:});
C = -Ainvm*Ainvm' - model.Ainv + Kuuinv;
for d = 1:model.nout,
    Kfdu = cell2mat( model.Kfu(d,:) ); 
    dF1_EZ = Kfdu.*(model.beta(d)*model.m{d}*Ainvm');
    dF_EZ2 = C.*(model.beta(d)*(Kfdu'*Kfdu));
    EZ = model.etadq(d,model.indXu)';
    for q = 1:model.nlf,
        EZt = EZ;
        indq = find(model.indXu==q);
        EZt(indq) = .5;
        getat(d,q) = sum(sum(dF1_EZ(:,indq))) + sum(sum( dF_EZ2(:,indq).*repmat(EZt, 1,length(indq)) ))...
                     - .5*model.beta(d)*sum(model.Kff{d,q});
        getat(d,q) = getat(d,q) + model.Elogpiq(q) - model.Elog1mProdpim(q)...
                     + log(1-model.etadq(d,q) + (model.etadq(d,q)==1))...
                     - log(model.etadq(d,q) + (model.etadq(d,q)==0));
    end
end

geta = reshape(getat,1,length(geta));