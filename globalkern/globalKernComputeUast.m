function Kuu = globalKernComputeUast(kern, outX, latX, gamma)

% LFMGLOBALKERNCOMPUTE
%
% COPYRIGTH : Mauricio A. Alvarez, 2013.
% MODIFICATIONS: Cristian Guarnizo, 2014.
% MULTIGP

if nargin < 4
    gamma = [];
end

Kuu = cell(kern.nlf,1);

kernLat = kern.template.latent;

% Compute Kuu -> rbf kernel
for k = 1:kern.nlf,
    % First we need to expand the parameters in the vector to the local
    % kernel
    if strcmp(kern.type(1),'g')
        kernLat.precisionU = kern.precisionU(k);
    else
        kernLat.inverseWidth = kern.inverseWidth(k);
    end
    Kuu{k} = real(kern.funcNames.computeLat(kernLat, outX{k}, latX{k}));
    if ~isempty(gamma),
        Kuu{k} = Kuu{k} + gamma(k)*eye(size(Kuu{k}));
    end
end
