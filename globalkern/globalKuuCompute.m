function Kuu = globalKuuCompute(kern,latX,q)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP

kernLat = globalSetKernLat(kern,q);
% Compute Kuu
Kuu = kern.funcNames.computeLat(kernLat, latX);