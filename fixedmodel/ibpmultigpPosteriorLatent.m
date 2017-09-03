function [umean, uvar] = ibpmultigpPosteriorLatent(model, Xtest)

Xlattest = cell(model.nlf,1);

for q=1:model.nlf,
    Xlattest{q} = Xtest;
end

if isfield(model, 'gamma') && ~isempty(model.gamma)
    Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX, model.gamma);
    %Kusus = globalKernComputeUast(model.kern, Xlattest, Xlattest, model.gamma);
else
    Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX);
    %Kusus = globalKernComputeUast(model.kern, Xlattest, Xlattest);
end


% fhandle = str2func([model.kernType 'KernCompute']);
% if isfield(model, 'gamma') && ~isempty(model.gamma)
%     [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
%         model.outX, model.latX, model.gamma);
% else
%     [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
%         model.outX, model.latX);
% end
% Psi2 = cell(1, model.nout);
% Psi1 = cell(model.nout, model.nlf);
% P = 0.;
% m2 = cell(model.nlf,1);
% for d = 1:model.nout,
%     EZ2 = model.etadq(d,:)'*model.etadq(d,:) - diag(model.etadq(d,:).^2)...
%         + diag(model.etadq(d,:));
%     Kfdu = cell2mat( model.Kfu(d,:) );
%     Psi2{d} = Kfdu'*Kfdu; %Q.MxQ.M
%     for q = 1:model.nlf,
%         if d == 1,
%             m2{q} = 0.;
%         end
%         Psi1{d,q} = model.Kfu{d,q}'*model.y{d}; %Mx1
%         m2{q} = m2{q} + (model.beta(d)*model.etadq(d,q))*Psi1{d,q}; %Mx1
%     end
%     P = P + model.beta(d)*(EZ2(model.indXu,model.indXu).*Psi2{d}); %Q.MxQ.M
% end
% model.m2 = cell2mat(m2);
% A = blkdiag(model.Kuu{:}) + P;  %Q.MxQ.M
% [La, jitter] = jitChol(A);
% if jitter > 1.,
%     error('Jitter higher than tolerance.')
% end
% Lainv = La\eye(size(La));
% model.Ainv = Lainv*Lainv.';


umean = cell(model.nlf,1);
uvar = cell(model.nlf,1);
for q=1:model.nlf,
    indq = model.indXu == q;
    umean{q} = Kusu{q}*(model.Ainv(indq,:)*model.m2);
    uvar{q} = 1. - diag(Kusu{q}*(model.Kuuinv{q}-model.Ainv(indq,indq))*Kusu{q}');
end


% fhandle = str2func([model.kernType 'KernCompute']);
% if isfield(model, 'gamma') && ~isempty(model.gamma)
%     [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
%         model.outX, model.latX, model.gamma);
% else
%     [model.Kff, model.Kfu, model.Kuu] = fhandle(model.kern, ...
%         model.outX, model.latX);
% end
% 
% if isfield(model, 'gamma') && ~isempty(model.gamma)
%     Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX, model.gamma);
% else
%     Kusu = globalKernComputeUast(model.kern, Xlattest, model.latX);
% end
% 
% A = cell(model.nlf,1);
% umean = cell(model.nlf,1);
% uvar = cell(model.nlf,1);
% Ainv = cell(model.nlf,1);
% for q=1:model.nlf,
%     T=0; %\bar{K}_{u_q,u_q}
%     Ut = 0;
%     for d=1:model.nout,
%         Ut = Ut + (model.etadq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.m{d});
%         T = T + (model.etadq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
%         %Ut = Ut + (model.etadq(d,q)*model.muSdq(d,q)*model.beta(d))*(model.Kfu{d,q}'*model.m{d});
%         %T = T + (model.etadq(d,q)*(model.muSdq(d,q)^2+model.varSdq(d,q))*model.beta(d))*(model.Kfu{d,q}'*model.Kfu{d,q});
%     end
%     
%     
%     A{q} = model.Kuu{q} + T; % A_{q,q} = K_{u_q,u_q} + \bat{K}_{u_q,u_q}
% 
%     Ainv{q} = pdinv(A{q});
%     T = Kusu{q}*Ainv{q};
%     umean{q} = T*Ut;
%     uvar{q} = diag(T*Kusu{q}');
% end