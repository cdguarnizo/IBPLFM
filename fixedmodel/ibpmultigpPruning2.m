function [eta, S] = ibpmultigpPruning2(Euast, eta_old, S_old)
Q = size(Euast,1);
eta = eta_old;
S = S_old;
for k = 1:Q,
    for l = k+1:Q,
        cr = corrcoef(Euast{k},Euast{l});
        cr = cr(1,2);
        if abs(cr)>=.99,
            S(:,k) = S_old(:,k) + sign(cr)*S_old(:,l);
            eta(:,l) = 1e-6;
            S(:,l) = 1e-6;
        end
    end
end