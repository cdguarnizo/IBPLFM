function [ymean, yvar, model] = ibpmultigpPosterior(model, Xtest)

ymean = cell(1, model.nout);
yvar = cell(1, model.nout);

fhandle = str2func([model.kernType 'KernCompute']);
if isfield(model, 'gamma') && ~isempty(model.gamma)
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX, model.gamma);
else
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX);
end

A = cell(model.nlf,1);
Ainv = cell(model.nlf,1);
for q =1 :model.nlf
    T = 0; %\bar{K}_{u_q,u_q}
    if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
        for d=1:model.nout
            %T = T + (model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
            T = T + (model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
        end
    else
        for d=1:model.nout
            T = T + ((model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
        end
    end
    if isfield(model, 'gamma') && ~isempty(model.gamma)
        A{q} = model.KuuGamma{q} + T; % A_{q,q} = K_{u_q,u_q} + \tilde{K}_{u_q,u_q} 
    else
        A{q} = model.Kuu{q} + T;
    end
    Ainv{q} = pdinv(A{q});
end

model.yhatdq = cell(model.nout,model.nlf);
for d = 1:model.nout %for each output
    for q = 1:model.nlf
        yhatdq = 0;
        
        %Yhat evaluation
        k=1:model.nlf;
        k(q) = [];
        
        if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
            for q2 = k
                yhatdq = yhatdq + model.etadq(d,q2)*model.muSdq(d,q2)*...
                    model.Kfu{d,q2}*model.Kuuinv{q2}*model.Euast{q2}; %TODO: When was calculted Kuuinv?
            end
        else
            for q2 = k
                yhatdq = yhatdq + model.muSdq(d,q2)*...
                    model.Kfu{d,q2}*model.Kuuinv{q2}*model.Euast{q2};
            end
        end
	model.yhatdq{d,q} = yhatdq;
    end
end

for d = 1:model.nout %for each output
    ymean{d} = 0;
    yvar{d} = 0;
    for q = 1:model.nlf
        alpha = 0;
        if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
            for d1=1:model.nout %hat{alpha} evaluation
                alpha = alpha + (model.etadq(d1,q)*model.muSdq(d1,q)*model.beta(d1))*(model.Kfu{d1,q}.'*(model.m{d1}-model.yhatdq{d1,q}));                
            end
            ymean{d} = ymean{d} + model.etadq(d,q)*model.muSdq(d,q)*Kfu{d,q}*Ainv{q}*alpha;
            yvar{d} = yvar{d} + model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*(diag(Kff{d,q})...
                - Kfu{d,q}*(model.Kuuinv{q}-Ainv{q})*Kfu{d,q}.');
        else
            for d1=1:model.nout %hat{alpha} evaluation
                alpha = alpha + (model.muSdq(d1,q)*model.beta(d1))*(model.Kfu{d1,q}.'*(model.m{d1}-model.yhatdq{d1,q}));
            end
            ymean{d} = ymean{d} + model.muSdq(d,q)*Kfu{d,q}*Ainv{q}*alpha;
            yvar{d} = yvar{d} + (model.muSdq(d,q)^2+model.varSdq(d,q))*(diag(Kff{d,q})...
                - Kfu{d,q}*(model.Kuuinv{q}-Ainv{q})*Kfu{d,q}');
        end
    end
    yvar{d} = diag(yvar{d})+1/model.beta(d);
    if isfield(model,'scale') && isfield(model,'bias')
        ymean{d} = ymean{d}*model.scale(d) + model.bias(d);
        yvar{d} = yvar{d}*model.scale(d)*model.scale(d);
    end
end