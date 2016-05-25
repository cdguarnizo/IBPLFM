function [ymean, yvar, model] = ibpmultigpPosterior(model, Xtest)

ymean = cell(1, model.nout);
yvar = cell(1, model.nout);

fhandle = str2func([model.kernType 'KernCompute']);
if isfield(model, 'gamma') && ~isempty(model.gamma)
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX, model.gamma);
else
    [Kff, Kfu, ~] = fhandle(model.kern, Xtest, model.latX);
end

if model.isVarU,
    
    A = cell(model.nlf,1);
    Ainv = cell(model.nlf,1);
    for q=1:model.nlf,
        T=0; %\bar{K}_{u_q,u_q}
        if model.isVarS,
            if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
                for d=1:model.nout,
                    T = T + (model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
                end
            else
                for d=1:model.nout,
                    T = T + ((model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
                end
            end
        else
            for d=1:model.nout,
                T = T + (model.etadq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
            end
        end
        A{q} = model.Kuu{q} + T; % A_{q,q} = K_{u_q,u_q} + \tilde{K}_{u_q,u_q}
        Ainv{q} = pdinv(A{q});
    end
    
    model.yhatdq = cell(model.nout,model.nlf);
    for d=1:model.nout, %for each output
        for q=1:model.nlf,
            yhatdq = 0;
            
            %Yhat evaluation
            k=1:model.nlf;
            k(q) = [];
            if model.isVarS,
                if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
                    for q2 = k,
                        yhatdq = yhatdq + model.etadq(d,q2)*model.muSdq(d,q2)*...
                            model.Kfu{d,q2}*model.Kuuinv{q2}*model.Euast{q2};
                    end
                else
                    for q2 = k,
                        yhatdq = yhatdq + model.muSdq(d,q2)*...
                            model.Kfu{d,q2}*model.Kuuinv{q2}*model.Euast{q2};
                    end
                end
            else
                for q2 = k,
                    yhatdq = yhatdq + model.etadq(d,q2)*...
                        model.Kfu{d,q2}*model.Kuuinv{q2}*model.Euast{q2};
                end
            end
            
            model.yhatdq{d,q} = yhatdq;
        end
    end

    for d=1:model.nout, %for each output
        ymean{d} = 0;
        yvar{d} = 0;
        for q=1:model.nlf,
            alpha = 0;
            if model.isVarS,
                if strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes')
                    for d1=1:model.nout, %hat{alpha} evaluation
                        alpha = alpha + (model.etadq(d1,q)*model.muSdq(d1,q)*model.beta(d1))*(model.Kfu{d1,q}.'*(model.m{d1}-model.yhatdq{d1,q}));
                    end
                    ymean{d} = ymean{d} + model.etadq(d,q)*model.muSdq(d,q)*Kfu{d,q}*Ainv{q}*alpha;
                    yvar{d} = yvar{d} + model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*(diag(Kff{d,q})...
                        - Kfu{d,q}*(model.Kuuinv{q}-Ainv{q})*Kfu{d,q}.');
                else
                    for d1=1:model.nout, %hat{alpha} evaluation
                        alpha = alpha + (model.muSdq(d1,q)*model.beta(d1))*(model.Kfu{d1,q}.'*(model.m{d1}-model.yhatdq{d1,q}));
                    end
                    ymean{d} = ymean{d} + model.muSdq(d,q)*Kfu{d,q}*Ainv{q}*alpha;
                    yvar{d} = yvar{d} + (model.muSdq(d,q)^2+model.varSdq(d,q))*(diag(Kff{d,q})...
                        - Kfu{d,q}*(model.Kuuinv{q}-Ainv{q})*Kfu{d,q}');
                end
            else
                for d1=1:model.nout, %hat{alpha} evaluation
                    alpha = alpha + (model.etadq(d1,q)*model.beta(d1))*(model.Kfu{d1,q}.'*(model.m{d1}-model.yhatdq{d1,q}));
                end
                ymean{d} = ymean{d} + model.etadq(d,q)*Kfu{d,q}*Ainv{q}*alpha;
                yvar{d} = yvar{d} + model.etadq(d,q)*(diag(Kff{d,q})...
                    - Kfu{d,q}*(model.Kuuinv{q}-Ainv{q})*Kfu{d,q}.');
            end
        end
        yvar{d} = diag(yvar{d})+1/model.beta(d);
        if isfield(model,'scale') && isfield(model,'bias'),
            ymean{d} = ymean{d}*model.scale(d) + model.bias(d);
            yvar{d} = yvar{d}*model.scale(d)*model.scale(d);
        end
    end
else
    MKfu = cell2mat(Kfu);    
    ys = MKfu*(model.A\model.m2);
    yvs = - MKfu*(blkdiag(model.Kuuinv{:}) - model.Ainv )*MKfu';
    ini = 0;
    for d = 1:model.nout,
        fin = ini + size(Xtest{d},1);
        ymean{d} = ys(ini+1:fin);
        yvar{d} = sum(sum(cell2mat(Kff(d,:)),2)) + diag(yvs(ini+1:fin, ini+1:fin)) + 1/model.beta(d);
        ini = fin;
        
        if isfield(model,'scale') && isfield(model,'bias'),
            ymean{d} = ymean{d}*model.scale(d) + model.bias(d);
            yvar{d} = yvar{d}*model.scale(d)*model.scale(d);
        end
        
    end
    
end