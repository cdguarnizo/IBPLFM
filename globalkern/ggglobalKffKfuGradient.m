function gParam = ggglobalKffKfuGradient(kern, outX, latX, dLdKyy, dLdKyu, d, q)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2017

gParam = zeros(1,kern.nout + kern.nlf + kern.nlf*kern.nout);

kernLat = kern.template.latent;
kernOut = kern.template.output;
%For precision U, Requires: dLdKyy, dLdKuy, dLdKuu
%For precision G, Requires: dLdKyy, dLdKuy

%gParam [precU precG X]
%gParam = zeros(1,kern.nout + kern.nlf);

%For precision G
%Requires: dLdKyy, dLdKuy

kernOut.precisionG = kern.precisionG(d);

kernOut.sensitivity = kern.sensitivity(d,q);
kernOut.precisionU = kern.precisionU(q);
kernLat.precisionU = kern.precisionU(q);

ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
g = real(kern.funcNames.gradientOut(kernOut, outX, dLdKyy)); %Kyygradient
gParam(ind) = gParam(ind) + g([1,2,4]);

g = real(kern.funcNames.gradientCross(kernOut, kernLat, outX, latX, dLdKyu)); %Kyu gradient
gParam(ind) = gParam(ind) + g([1,2,4]);

if kern.options.isVarS,
    gParam(end-kern.nlf*kern.nout+1:end) = [];
end
