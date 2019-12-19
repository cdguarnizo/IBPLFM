function model = ibpmultigpComputeELog(model)

% IBPMULTIGPCOMPUTEELOG

% IBPMULTIGP

for q = 1:model.nlf
    % Compute E[log pi_q]
    model.Elogpiq(q) = sum(psi(model.tau1(1:q)) - ...
        psi(model.tau1(1:q)+model.tau2(1:q)));
    % Compute E[log(1- prod(vm))]
    qqnsum = fliplr(cumsum(fliplr(model.qki{q})));
    model.Elog1mProdpim(q) = sum(model.qki{q}.*psi(model.tau2(1:q))) ...
        - sum(qqnsum.*psi(model.tau1(1:q) +model.tau2(1:q))) ...
        - sum(model.qki{q}.*log(model.qki{q}));
    if q > 1
    model.Elog1mProdpim(q) = model.Elog1mProdpim(q) + ...
        sum(qqnsum(2:end).*psi(model.tau1(1:q-1)));
    end
end