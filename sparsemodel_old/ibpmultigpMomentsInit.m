function model = ibpmultigpMomentsInit(model)

% IBPMULTIGPMOMENTSINIT

% IBPMULTIGP
if strcmp(model.sparsePriorType,'ibp')
    % Initialize tau
    model.tau1 = randi(3, 1, model.nlf);
    model.tau2 = randi(3, 1, model.nlf);
    % Initialize qki
    for q=1:model.nlf
        model.qki{q} = rand(1, q);
        model.qki{q} = model.qki{q}/sum(model.qki{q});
    end
    % Compute expectations difficult terms IBP
    model = ibpmultigpComputeELog(model);
elseif strcmp(model.sparsePriorType,'spikes')
    model.pi = 0.5;
end

% Moments for q(uq)
for q=1:model.nlf
   model.Euast{q, 1} = rand(model.k, 1);
   model.Kuuast{q, 1} = eye(model.k);
end
% model.Euuast = cell(model.nlf);
% for q1=1:model.nlf      
%    model.Euuast{q1, q1} = eye(model.k) + model.Euast{q1}*model.Euast{q1}';
%    for q2 = 1:q1-1
%        model.Euuast{q1, q2} = model.Euast{q1}*model.Euast{q2}';
%        model.Euuast{q2, q1} = model.Euuast{q1, q2}'; 
%    end
% end

if model.gammaPrior
    % Moments for q(gammadq) 
    model.adqast = randi(5, model.nout, model.nlf);
    model.bdqast = randi(5, model.nout, model.nlf);
end
    
%load yeastvar;
if model.Trainvar
    if model.pso
        %find initial good solution
        model.varSdq = rand(model.nout, model.nlf);
        [S, LB] = pso(model);
        model.etadq = reshape(S(1:model.nlf*model.nout),model.nout,model.nlf);
        model.muSdq = reshape(S(model.nlf*model.nout+1:2*model.nlf*model.nout),model.nout,model.nlf);
    else
        if (strcmp(model.sparsePriorType,'ibp') || strcmp(model.sparsePriorType,'spikes'))
            % Initialize etadq (moment for q(Zdq)) DIDIT as Titsias does
            %model.etadq = zeros(model.nout, model.nlf) + 0.01;
            model.etadq = 0.5*ones(model.nout, model.nlf) + 0.01*randn(model.nout, model.nlf);
            %model.etadq(model.etadq < 0.2) = 0.2;
            %model.etadq(model.etadq > 0.8) = 0.8;
            
            %model.etadq = 0.9*ones(model.nout, model.nlf);
            %model.etadq = rand(model.nout, model.nlf);
        end
        % Initialize muSdq and varSdq (moments for q(Sdq|Zdq) )
        model.muSdq = randn(model.nout, model.nlf);
        model.varSdq = 0.1*ones(model.nout, model.nlf);
    end
    % % model.muSdq = ones(model.nout, model.nlf);
    % % model.varSdq = 0.1*ones(model.nout, model.nlf);
    % model.ES2dq = model.varSdq + model.muSdq.^2;
    % % Moments for q(Sdq, Zdq)
    % model.EZdqSdq = model.etadq.*model.muSdq;
    % model.EZdqS2dq = model.etadq.*(model.varSdq + model.muSdq.^2);
    % model.VZdqS2dq = model.EZdqS2dq - model.EZdqSdq.^2;
    % model.StdZdqS2dq = sqrt(model.VZdqS2dq);
end

if ~model.gammaPrior
    model.gammadq = model.muSdq.^2 + model.varSdq;
end