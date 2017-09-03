function gParam = globalKffKfuGradient(kern, outX, latX, dLdKyy, dLdKyu, d, q)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2017

%For precision U, Requires: dLdKyy, dLdKuy, dLdKuu
%For precision G, Requires: dLdKyy, dLdKuy

%gParam [precU precG X]
gParam = zeros(1,kern.nParams);

kernLat = globalSetKernLat(kern,q);
kernOut = globalSetKernOut(kern,d,q);

switch kernOut.type,
    case 'gg'
        indf = [1 2 4];
        ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
    case 'sim'
        indf = [2 1 3];
        ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
    case 'lfm'
        if kern.incMass,
            indf = [4 1 2 3 5];
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d, kern.nlf + kern.nout + d,...
                kern.nlf + 2*kern.nout + d + (q-1)*kern.nout];
        else
            indf = [4 2 3 5];
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d, ...
                kern.nlf + 2*kern.nout + d + (q-1)*kern.nout];
        end
end
if kern.options.isVarS,
    indf(end) = [];
    ind(end) = [];
end

g = real(kern.funcNames.gradientOut(kernOut, outX, dLdKyy)); %Kyygradient
gParam(ind) = gParam(ind) + g(indf);

g = real(kern.funcNames.gradientCross(kernOut, kernLat, outX, latX, dLdKyu)); %Kyu gradient
gParam(ind) = gParam(ind) + g(indf);