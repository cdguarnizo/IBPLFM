function [dLdKyy, dLdKuy, dLdKuu, dLdmu, dLdbeta] = ibpmultigpLocalCovGradients(model)

% IBPMULTIGPLOCALCOVGRADIENTS

% IBPMULTIGP

dLdKuy = cell(model.nlf,1);
startLat = 1;
endLat = 0;
for r =1:model.nlf,
    endLat = endLat + model.k;
    startOut = 1;
    endOut = 0;
    for k= 1:model.nout,
        endOut = endOut + model.sizeX(k);
        dLdKuy{r,k} = model.KuuinvKuy(startLat:endLat, startOut:endOut)...
            *model.beta(k)*model.exS2(k,r);
        startOut = endOut + 1;
    end
    startLat = endLat + 1;
end
dLdKuu = cell(model.nlf,1);
startLat = 1;
endLat = 0;
for r =1:model.nlf,
    endLat = endLat + model.k;
    KuuinvKuyQKyuKuuinv = zeros(model.k);
    dLdKuu{r} = zeros(model.k);
    startOut = 1;
    endOut = 0;
    for k=1: model.nout,
        endOut = endOut + model.sizeX(k);
        KuuinvKuy = model.KuuinvKuy(startLat:endLat, startOut:endOut);
        KuuinvKuyQKyuKuuinv =  KuuinvKuyQKyuKuuinv + ...
            KuuinvKuy*KuuinvKuy'*model.beta(k)*model.exS2(k,r);
        startOut = endOut + 1;
    end
    dLdKuu{r} = - 0.5*(model.Kuuinv{r} + KuuinvKuyQKyuKuuinv);
    startLat = endLat + 1;
end
dLdbeta = zeros(1,model.nout);
dLdmu = cell(model.nout,1);
dLdKyy = cell(model.nout,model.nlf);
for k =1:model.nout,
    endOut = endOut + model.sizeX(k);
    temp = kron(-0.5*model.beta(k)*model.exS2(k,:), ...
        sparseDiag(ones(model.sizeX(k),1)));
    dLdKyy(k,:) = mat2cell(temp, model.sizeX(k), ...
        model.sizeX(k)*ones(1, model.nlf));
    dLdmu{k} = model.beta(k)*model.m{k};
    dLdbeta(k) = -0.5*(model.Ktilde{k} - model.sizeX(k)/model.beta(k) ...
        + model.m{k}'*model.m{k});
end