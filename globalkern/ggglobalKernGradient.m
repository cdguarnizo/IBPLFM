function gParam = ggglobalKernGradient(kern, outX, latX, dLdKyy, dLdKyu, dLdKuu)

% GGGLOBALKERNGRADIENT
%
% COPYRIGTH : Cristian Guarnizo, 2014

gParam = zeros(1,kern.nout + kern.nlf + kern.nlf*kern.nout);
if strcmp(kern.approx,'ftc'),
    %For FTC it is only required dLdKyy
    kernOut = kern.template.output;
    kernOut2 = kern.template.output;
    for q = 1:kern.nlf,
        kernOut.precisionU = kern.precisionU(q);
        kernOut2.precisionU = kern.precisionU(q);
        for d = 1:kern.nout,
            kernOut.precisionG = kern.precisionG(d);
            kernOut.sensitivity = kern.sensitivity(d,q);
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
            %[PrecU(1xnlf) PrecG(1xnout) Sensititvities(noutxnlf)]
            g = .5*real(kern.funcNames.gradientOut(kernOut, outX{d}, dLdKyy{d,d})); %Kyygradient
            %Add gradients
            gParam(ind) = gParam(ind) + g([1,2,4]); % PrecisionU PrecisionG Sensitivity
            for dp = d+1:kern.nout,
                kernOut2.precisionG = kern.precisionG(dp);
                kernOut2.sensitivity = kern.sensitivity(dp,q);
                [g1, g2] = kern.funcNames.gradientCrossOut(kernOut, kernOut2, outX{d}, outX{dp}, dLdKyy{d,dp}); %Kyygradient
                %Add gradients
                gParam(ind) = gParam(ind) + g1([1,2,4]); % PrecisionU PrecisionG Sensitivity
                ind2 = [kern.nlf + dp, kern.nlf + kern.nout + dp + (q-1)*kern.nout];
                gParam(ind2) = gParam(ind2) + g2([2,4]); % PrecisionG Sensitivity
            end
        end
    end
    if kern.options.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
else
    kernLat = kern.template.latent;
    kernOut = kern.template.output;
    %For precision U, Requires: dLdKyy, dLdKuy, dLdKuu
    %For precision G, Requires: dLdKyy, dLdKuy
    
    %gParam [precU precG X]
    %gParam = zeros(1,kern.nout + kern.nlf);
    
    for k = 1:kern.nlf,
        kernLat.precisionU = kern.precisionU(k);
        %gaussianXgaussian
        g = real(kern.funcNames.gradientLat(kernLat, latX{k}, dLdKuu{k})); %Kuugradient
        gParam(1,k) = g(1); %Kuugradient
    end
    
    %For precision G
    %Requires: dLdKyy, dLdKuy
    for d = 1:kern.nout,
        kernOut.precisionG = kern.precisionG(d);
        for q = 1:kern.nlf,
            kernOut.sensitivity = kern.sensitivity(d,q);
            kernOut.precisionU = kern.precisionU(q);
            kernLat.precisionU = kern.precisionU(q);
            
            ind = [q, kern.nlf + d, kern.nlf + kern.nout + d + (q-1)*kern.nout];
            g = real(kern.funcNames.gradientOut(kernOut, outX{d}, dLdKyy{d,q})); %Kyygradient
            gParam(ind) = gParam(ind) + g([1,2,4]);
            
            g = real(kern.funcNames.gradientCross(kernOut, kernLat, outX{d}, latX{q}, dLdKyu{d,q})); %Kyu gradient
            gParam(ind) = gParam(ind) + g([1,2,4]);
        end
    end
    if kern.options.isVarS,
        gParam(end-kern.nlf*kern.nout+1:end) = [];
    end
end