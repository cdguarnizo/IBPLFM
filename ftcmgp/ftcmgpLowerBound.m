function logp = ftcmgpLowerBound(model)
% FTCMGPLOWERBOUND
% FTCMGP
if iscell(model.y),
    yvec = cell2mat(model.y);
else
    yvec = model.y;
end
logp = -0.5*(yvec'*model.alpha + length(yvec)*log(2*pi)) - sum(log(diag(model.L)));