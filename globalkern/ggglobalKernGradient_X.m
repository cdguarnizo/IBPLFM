function gParam = ggglobalKernGradient_X(kern, outX, latX, dLdKyu, dLdKuu)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2014


kernLat = kern.template.latent;
kernOut = kern.template.output;

%We assume that each latent input has the same number of inducing points
width = length(latX{1}(:));
gParam = zeros(1,kern.nlf*width);

for k = 1:kern.nlf,
    kernLat.precisionU = kern.precisionU(k);
    gParam(1,1+(k-1)*width:k*width) = gaussianKernGradient_X(kernLat, latX{k}, dLdKuu{k}); %Kuugradient
end

% Requires: dLdKyu
for i = 1:kern.nout,
    kernOut.precisionG = kern.precisionG(i);
    for j = 1:kern.nlf,
        kernOut.precisionU = kern.precisionU(j);
        kernLat.precisionU = kern.precisionU(j);
        g = ggXgaussianKernGradient_X(kernOut, kernLat, outX{i}, latX{j}, dLdKyu{i,j});
        gParam(1,1+(j-1)*width:j*width) = gParam(1,1+(j-1)*width:j*width) + g;
    end
end