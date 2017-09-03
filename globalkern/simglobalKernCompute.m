function [Kff, Kfu, Kuu] = simglobalKernCompute(kern, outX, latX, gamma)

% LFMGLOBALKERNCOMPUTE
%
% COPYRIGTH : Mauricio A. Alvarez, 2013.
% MODIFICATIONS: Cristian Guarnizo, 2014.
% MULTIGP

if nargin < 4
    gamma = [];
end

if strcmp(kern.approx,'ftc'),
    %Kff = zeros(length(outX), length(outX));
    if nargin < 3,
        Kfft = cell(kern.nout, kern.nout);
        Kff = cell(1,kern.nlf);
        kernOut = kern.template.output;
        kernOut2 = kern.template.output;
        for q = 1:kern.nlf,
            kernOut.inverseWidth = kern.inverseWidth(q);
            kernOut2.inverseWidth = kern.inverseWidth(q);
            for d = 1:kern.nout,
                kernOut.decay = kern.decay(d);
                kernOut.sensitivity = kern.sensitivity(d,q);
                Kfft{d,d} = real(kern.funcNames.computeOut(kernOut, outX{d}));
                for dp = d+1:kern.nout,
                    kernOut2.decay = kern.decay(dp);
                    kernOut2.sensitivity = kern.sensitivity(dp,q);
                    Kfft{d,dp} = real(kern.funcNames.computeCrossOut(kernOut, kernOut2, outX{d}, outX{dp}));
                    Kfft{dp,d} = Kfft{d,dp}.';
                end
            end
            Kff{q} = cell2mat(Kfft);
            %Kff = Kff + Kfu{q};
        end
    else
        Kfft = cell(kern.nout, kern.nout);
        Kff = cell(1,kern.nlf);
        kernOut = kern.template.output;
        kernOut2 = kern.template.output;
        for q = 1:kern.nlf,
            kernOut.inverseWidth = kern.inverseWidth(q);
            kernOut2.inverseWidth = kern.inverseWidth(q);
            for d = 1:kern.nout,
                kernOut.decay = kern.decay(d);
                kernOut.sensitivity = kern.sensitivity(d,q);
                Kfft{d,d} = real(kern.funcNames.computeOut(kernOut, outX{d}, latX{d}));
                for dp = d+1:kern.nout,
                    kernOut2.decay = kern.decay(dp);
                    kernOut2.sensitivity = kern.sensitivity(dp,q);
                    Kfft{d,dp} = real(kern.funcNames.computeCross(kernOut, kernOut2, outX{d}, latX{dp}));
                    Kfft{dp,d} = Kfft{d,dp}.';
                end
            end
            Kff{q} = cell2mat(Kfft);
            %Kff = Kff + Kfu{q};
        end
    end
else
    
    Kuu = cell(kern.nlf,1);
    Kfu = cell(kern.nout, kern.nlf);
    Kff = cell(kern.nout, kern.nlf);
    
    kernLat = kern.template.latent;
    kernLat.isNegativeS = true;
    kernOut = kern.template.output;
    kernOut.isNegativeS = true;
    % Compute Kuu -> rbf kernel
    for k = 1:kern.nlf,
        % First we need to expand the parameters in the vector to the local
        % kernel
        kernLat.inverseWidth = kern.inverseWidth(k);
        Kuu{k} = real(kern.funcNames.computeLat(kernLat, latX{k}));
        if ~isempty(gamma),
            Kuu{k} = Kuu{k} + gamma(k)*eye(size(Kuu{k}));
        end
    end

    for d = 1:kern.nout,
        %Here expand spring and damper
        kernOut.decay = kern.decay(d);
        for q = 1:kern.nlf,
            kernOut.sensitivity = kern.sensitivity(d,q);
            % Expand the parameter inverseWidth
            kernOut.inverseWidth = kern.inverseWidth(q);
            kernLat.inverseWidth = kern.inverseWidth(q);
            % Compute Kff
            Kff{d,q} = real(kern.funcNames.computeOut(kernOut, outX{d}));
            
            if any(isinf(Kff{d,q})) | any(isnan(Kff{d,q})),
                error('Nan or Inf in Kff')
            end
            
            % Compute Kfu, which corresponds to K_{\hat{f}}u
            Kfu{d,q} = real(kern.funcNames.computeCross(kernOut, kernLat, outX{d}, latX{q}));
            
            if any(isinf(Kfu{d,q})) | any(isnan(Kfu{d,q})),
                error('Nan or Inf in Kfu')
            end
        end
    end
end
