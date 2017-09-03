function model = ftcmgpComputeKernel(model)

% FTCMGPCOMPUTEKERNELS
% FTCMGP

fhandle = str2func([model.kernType 'KernCompute']);
Kffq = fhandle(model.kern, ind2cell(model.outX.val, model.outX.ind));

model.Kyy = zeros(size(model.y,1));

if model.isVarS,
    for k = 1:model.nlf,
        SS = model.S(:, k)*model.S(:, k)';
        model.Kyy = model.Kyy + SS(model.outX.ind, model.outX.ind).*Kffq{k};
    end
else
    for k = 1:model.nlf,
        model.Kyy = model.Kyy + Kffq{k};
    end
end

if model.includeNoise,
    % Generating noise vector
    
    if isfield(model,'lambdae') && ~isempty(model.lambdae),
        model.beta = model.lambdae;
    end
    
    noise = cell(model.nout,1);
    for d = 1:model.nout,
        noise{d} = zeros(length(model.y(model.outX.ind == d)),1);
        noise{d}(:) = 1/model.beta(d);
    end
    noise = cell2mat(noise);
    % Adding noise
    [row, col] = size(model.Kyy);
    diagind = 1:row+1:row*col;
    model.Kyy(diagind) = model.Kyy(diagind) + noise';
end
% Update alpha
model.L = jitChol(model.Kyy);
Li = model.L\eye(size(model.Kyy,1));
model.Kyyinv = Li*Li';
model.alpha =  model.Kyyinv*model.y;