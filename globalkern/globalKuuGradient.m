function gParam = globalKuuGradient(kern, latX, dLdKuu, q)


gParam = zeros(1,kern.nParams);

kernLat = globalSetKernLat(kern, q);
g = real(kern.funcNames.gradientLat(kernLat, latX, dLdKuu)); %Kuugradient
gParam(1,q) = g(1); %Kuugradient