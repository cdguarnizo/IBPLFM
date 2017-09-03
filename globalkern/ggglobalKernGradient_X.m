function gParam = ggglobalKernGradient_X(kern, outX, latX, dLdKyu, dLdKuu)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2014


kernLat = kern.template.latent;
kernOut = kern.template.output;

% We assume that each latent input has the same number of inducing points
width = length(latX{1}(:));
gParam = zeros(1,kern.nlf*width);

for k = 1:kern.nlf,
    kernLat.precisionU = kern.precisionU(k);
    gParam(1,1+(k-1)*width:k*width) = gaussianKernGradient_X(kernLat, latX{k}, dLdKuu{k}); %Kuugradient
end

% Requires: dLdKyu
for d = 1:kern.nout,
    kernOut.precisionG = kern.precisionG(d);
    for q = 1:kern.nlf,
        kernOut.precisionU = kern.precisionU(q);
        kernLat.precisionU = kern.precisionU(q);
        g = ggXgaussianKernGradient_X(kernOut, kernLat, outX{d}, latX{q}, dLdKyu{d,q});
        index = 1+(q-1)*width:q*width;
        gParam(1,index) = gParam(1,index) + g;
    end
end