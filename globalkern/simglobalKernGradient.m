function gParam = simglobalKernGradient(kern, outX, latX, dLdKyy, dLdKyu, dLdKuu)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Mauricio A. Alvarez, 2013
% MODIFICATIONS : Cristian Guarnizo, 2014

gParam = zeros(1,kern.nlf + kern.nlf*kern.nout + kern.nout);
if strcmp(kern.approx,'ftc'),
    %For FTC it is only required dLdKyy
    kernOut = kern.template.output;
    kernOut2 = kern.template.output;
    for q = 1:kern.nlf,
        kernOut.inverseWidth = kern.inverseWidth(q);
        kernOut2.inverseWidth = kern.inverseWidth(q);
        for d = 1:kern.nout,
            kernOut.decay = kern.decay(d);
            kernOut.sensitivity = kern.sensitivity(d,q);
            
            %inverseWidth decay sensititivity
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
            g = .5*real(kern.funcNames.gradientOut(kernOut, outX{d}, dLdKyy{d,d})); %Kyy gradient
            %Add gradients
            gParam(ind) = gParam(ind) + g([2 1 3]);
            for dp = d+1:kern.nout,
                kernOut2.decay = kern.decay(dp);
                kernOut2.sensitivity = kern.sensitivity(dp,q);
                [g1, g2] = kern.funcNames.gradientCross(kernOut, kernOut2, outX{d}, outX{dp}, dLdKyy{d,dp}); %Kyygradient
                %Add gradients
                gParam(ind) = gParam(ind) + g1([2 1 3]);
                ind2 = [kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
                gParam(ind2) = gParam(ind2) + g2([1 3]);
            end
        end
    end
    if kern.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
else
    kernLat = kern.template.latent;
    kern.isNegativeS = true;
    kernOut = kern.template.output;
    kernOut.isNegativeS = true;
    %For precision inverseWidth, Requires: dLdKyy, dLdKuy, dLdKuu
    
    %For precision decay, Requires: dLdKyy, dLdKuy
    
    %gParam [inverseWidth decay senstivity]
    %gParam = zeros(1,kern.nlf + kern.nout + kern.out*kern.nlf);
    
    for q = 1:kern.nlf,
        kernLat.inverseWidth = kern.inverseWidth(q);
        %gaussianXgaussian
        g = real(kern.funcNames.gradientLat(kernLat, latX{q}, dLdKuu{q})); %Kuugradient
        gParam(1,q) = g(1);
    end
    
    %For decay inverseWidth
    %Requires: dLdKyy, dLdKuy
    for d = 1:kern.nout,
        kernOut.decay = kern.decay(d);
        for q = 1:kern.nlf,
            kernOut.sensitivity = kern.sensitivity(d,q);
            kernOut.inverseWidth = kern.inverseWidth(q);
            kernLat.inverseWidth = kern.inverseWidth(q);
            %inverseWidth decay sensititivity
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
            
            g = real(kern.funcNames.gradientOut(kernOut, outX{d}, dLdKyy{d,q})); %Kyy gradient
            gParam(ind) = gParam(ind) + g([2 1 3]);
            
            g = real(kern.funcNames.gradientCross(kernOut, kernLat, outX{d}, latX{q}, dLdKyu{d,q})); %Kuy gradient
            gParam(ind) = gParam(ind) + g([2 1 3]);
        end
    end
    if kern.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
end