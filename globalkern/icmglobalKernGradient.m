function gParam = icmglobalKernGradient(kern, outX, latX, dLdKyy, dLdKyu, dLdKuu)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Mauricio A. Alvarez, 2013
% MODIFICATIONS : Cristian Guarnizo, 2014, 2015

gParam = zeros(1,kern.nlf + kern.nlf*kern.nout);
if strcmp(kern.approx,'ftc'),
    %For FTC it is only required dLdKyy
    kernOut = kern.template.output;
    kernOut2 = kern.template.output;
    for q = 1:kern.nlf,
        kernOut.inverseWidth = kern.inverseWidth(q);
        kernOut2.inverseWidth = kern.inverseWidth(q);
        for d = 1:kern.nout,
            kernOut.mass = kern.mass(d);
            kernOut.spring = kern.spring(d);
            kernOut.damper = kern.damper(d);
            kernOut.sensitivity = kern.sensitivity(d,q);
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d, ...
                kern.nlf + 2*kern.nout + d, kern.nlf + 3*kern.nout + d + (q-1)*kern.nout];
            %inverseWidth mass spring damper sensitivity
            g = .5*real(kern.funcNames.gradientOut(kernOut, outX{d}, dLdKyy{d,d})); %Kyy gradient
            %Add gradients
            gParam(ind) = gParam(ind) + g([4 1 2 3 5]);
            for dp = d+1:kern.nout,
                kernOut2.mass = kern.mass(dp);
                kernOut2.spring = kern.spring(dp);
                kernOut2.damper = kern.damper(dp);
                kernOut2.sensitivity = kern.sensitivity(dp,q);
                [g1, g2] = kern.funcNames.gradientCrossOut(kernOut, kernOut2, outX{d}, outX{dp}, dLdKyy{d,dp}); %Kyygradient
                %Add gradients
                gParam(ind) = gParam(ind) + g1([4 1 2 3 5]);
                ind2 = [kern.nlf + dp, kern.nlf + kern.nout + dp, ...
                kern.nlf + 2*kern.nout + dp, kern.nlf + 3*kern.nout + dp + (q-1)*kern.nout];
                gParam(ind2) = gParam(ind2) + g2([1 2 3 5]);
            end
        end
    end
    if ~kern.incMass,
        gParam(kern.nlf+1:kern.nlf+kern.nout) = [];
    end
    if kern.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
else
    kernLat = kern.template.latent;
    %For precision inverseWidth, Requires: dLdKyy, dLdKuy, dLdKuu
    
    %For precision Spring and Damper, Requires: dLdKyy, dLdKuy
    
    %gParam [inverseWidth mass Spring Damper]
    %gParam = zeros(1,kern.nlf + 3*kern.nout);
    
    for q = 1:kern.nlf,
        kernLat.inverseWidth = kern.inverseWidth(q);
        %gaussianXgaussian
        g = rbfKernGradient(kernLat, latX{q}, dLdKuu{q}); %Kuugradient
        gParam(1,q) = g(1);
    end
    
    %For mass, spring, damper and inverseWidth
    %Requires: dLdKyy, dLdKuy
    for d = 1:kern.nout,   
        for q = 1:kern.nlf,
            kernLat.inverseWidth = kern.inverseWidth(q);
            %inverseWidth mass spring damper
            ind = [q, kern.nlf + d + (q-1)*kern.nout];
            
            g = rbfKernGradient(kernLat, outX{d}, diag(dLdKyy{d,q})); %Kyy gradient
            g(2) = g(2)*2*kern.sensitivity(d,q);
            gParam(ind) = gParam(ind) + g([1 2]);
            
            g = rbfKernGradient(kernLat, outX{d}, latX{q}, dLdKyu{d,q}); %Kuy gradient
            gParam(ind) = gParam(ind) + g([1 2]);
        end
    end
    if kern.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
end