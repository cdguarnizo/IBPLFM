function [ymean, yvar] = ibpmultigpPosterior(model, Xtest)

ymean = cell(model.nout, 1);
yvar = cell(model.nout, 1);

fhandle = str2func([model.kernType 'KernCompute']);
if isfield(model, 'gamma') && ~isempty(model.gamma)
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX, model.gamma);
else
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX);
end

%if model.isVarU,

for d=1:model.nout, %for each output
    ymean{d} = 0;
    yvar{d} = 0;
    Ainv = mat2cell(model.Ainv, model.sizeXu, model.sizeXu);
    for q = 1:model.nlf,
        for k = [1:q-1,q+1:model.nlf],
            T = model.etadq(d,k)*(Ainv{q,k}*Kfu{d,k}.');
        end
        ymean{d} = ymean{d} + model.etadq(d,q)*Kfu{d,q}*model.Kuuinv{q}*model.Euast{q};
        yvar{d} = yvar{d} + model.etadq(d,q)*( diag(Kff{d,q})...
            - Kfu{d,q}*((model.Kuuinv{q}-Ainv{q,q})*Kfu{d,q}.' - T) );
    end
    if model.UseMeanConstants,
        ymean{d} = ymean{d} + model.mu(d);
    end
    yvar{d} = diag(yvar{d})+1/model.beta(d);
    if isfield(model,'scale') && isfield(model,'bias'),
        ymean{d} = ymean{d}*model.scale(d) + model.bias(d);
        yvar{d} = yvar{d}*model.scale(d)*model.scale(d);
    end
    
end

% else
%     MKfu = cell2mat(Kfu);    
%     ys = MKfu*(model.A\model.m2);
%     yvs = - MKfu*(blkdiag(model.Kuuinv{:}) - model.Ainv )*MKfu';
%     ini = 0;
%     for d = 1:model.nout,
%         fin = ini + size(Xtest{d},1);
%         ymean{d} = ys(ini+1:fin);
%         yvar{d} = sum(sum(cell2mat(Kff(d,:)),2)) + diag(yvs(ini+1:fin, ini+1:fin)) + 1/model.beta(d);
%         ini = fin;
%         
%         if isfield(model,'scale') && isfield(model,'bias'),
%             ymean{d} = ymean{d}*model.scale(d) + model.bias(d);
%             yvar{d} = yvar{d}*model.scale(d)*model.scale(d);
%         end
%         
%     end
%     
end