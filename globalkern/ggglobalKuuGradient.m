function gParam = ggglobalKuuGradient(kern, latX, dLdKuu, q)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2014

gParam = zeros(1,kern.nout + kern.nlf + kern.nlf*kern.nout);

kernLat = kern.template.latent;
%For precision U, Requires: dLdKyy, dLdKuy, dLdKuu
%For precision G, Requires: dLdKyy, dLdKuy

%gParam [precU precG X]
%gParam = zeros(1,kern.nout + kern.nlf);

kernLat.precisionU = kern.precisionU(q);
%gaussianXgaussian
g = real(kern.funcNames.gradientLat(kernLat, latX, dLdKuu)); %Kuugradient
gParam(1,q) = g(1); %Kuugradient

if kern.options.isVarS,
    gParam(end-kern.nlf*kern.nout+1:end) = [];
end
