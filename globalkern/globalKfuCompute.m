function Kfu = globalKfuCompute(kern, outX, latX, d, q)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP

% kernOut = kern.template.output;
% kernLat = kern.template.latent;
% % Expand the parameter decay
% kernOut.precisionG = kern.precisionG(d);
% kernOut.sensitivity = kern.sensitivity(d,q);
% % Expand the parameter inverseWidth
% kernOut.precisionU = kern.precisionU(q);
% kernLat.precisionU = kern.precisionU(q);

kernLat = globalSetKernLat(kern,q);
kernOut = globalSetKernOut(kern,d,q);

% Compute Kfu, which corresponds to K_{\hat{fu}}, really.
Kfu = real(kern.funcNames.computeCross(kernOut, kernLat, outX, latX));

if any(isnan(Kfu)) | any(isinf(Kfu)),
    error('Nan or Inf in Kfu')
end
