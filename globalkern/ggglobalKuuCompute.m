function Kuu = ggglobalKuuCompute(kern,latX,q)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP


kernLat = kern.template.latent;

% Compute Kuu

% First we need to expand the parameters in the vector to the local
% kernel
kernLat.precisionU = kern.precisionU(q);
Kuu = kern.funcNames.computeLat(kernLat, latX);