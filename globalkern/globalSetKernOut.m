function kOut = globalSetKernOut(kern,d,q)


kOut = kern.template.output;

switch kOut.type
    case 'gg'
        kOut.precisionU = kern.precisionU(q);
        kOut.precisionG = kern.precisionG(d);
        kOut.sensitivity = kern.sensitivity(d,q);
    case 'sim'
        kOut.inverseWidth = kern.inverseWidth(q);
        kOut.decay = kern.decay(d);
        kOut.sensitivity = kern.sensitivity(d,q);
    case 'lfm'
        kOut.inverseWidth = kern.inverseWidth(q);
        kOut.mass = kern.mass(d);
        kOut.damper = kern.damper(d);
        kOut.spring = kern.spring(d);
        kOut.sensitivity = kern.sensitivity(d,q);
end
