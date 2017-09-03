function kLat = globalSetKernLat(kern,q)

kLat = kern.template.latent;

switch kLat.type
    case 'gaussian'
        kLat.precisionU = kern.precisionU(q);
    case 'rbf'
        kLat.inverseWidth = kern.inverseWidth(q);
end