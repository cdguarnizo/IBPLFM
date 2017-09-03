function [eta, S] = ibpmultigpPruning(Euast, eta_old, S_old)
Q = size(Euast,1);
eta = eta_old;
S = S_old;
for k = Q:-1:2,
    Zk = round(eta(:,k));
    for l = k-1:-1:1,
        Zl = round(eta(:,l));
        cr = corrcoef(Euast{k},Euast{l});
        cr = cr(1,2);
        Sf = all( sign(S(:,k)).*Zk == sign(cr)*sign(S(:,l)).*Zl );
        if abs(cr)>=.99 && Sf,
            S(:,l) = S(:,l) + sign(cr)*S(:,k);
            eta(:,k) = 1e-3;
            S(:,k) = 0.;
        end
    end
end