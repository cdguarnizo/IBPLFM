function [Kff, Kfu, Kuu] = icmglobalKernCompute(kern, outX, latX, gamma)

% ICMGLOBALKERNCOMPUTE
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
                kernOut.mass = kern.mass(d);
                kernOut.damper = kern.damper(d);
                kernOut.spring = kern.spring(d);
                kernOut.sensitivity = kern.sensitivity(d,q);
                Kfft{d,d} = real(kern.funcNames.computeOut(kernOut, outX{d}));
                for dp = d+1:kern.nout,
                    kernOut2.mass = kern.mass(dp);
                    kernOut2.damper = kern.damper(dp);
                    kernOut2.spring = kern.spring(dp);
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
                kernOut.mass = kern.mass(d);
                kernOut.damper = kern.damper(d);
                kernOut.spring = kern.spring(d);
                kernOut.sensitivity = kern.sensitivity(d,q);
                Kfft{d,d} = real(kern.funcNames.computeOut(kernOut, outX{d}, latX{d}));
                for dp = d+1:kern.nout,
                    kernOut2.mass = kern.mass(dp);
                    kernOut2.damper = kern.damper(dp);
                    kernOut2.spring = kern.spring(dp);
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
    
    % Compute Kuu -> rbf kernel
    for k = 1:kern.nlf,
        % First we need to expand the parameters in the vector to the local
        % kernel
        kernLat.inverseWidth = kern.inverseWidth(k);
        Kuu{k} = rbfKernCompute(kernLat, latX{k});
        if ~isempty(gamma) %Ask Mauricio about this conditional
            Kuu{k} = Kuu{k} + gamma(k)*eye(size(Kuu{k}));
        end
    end
    for d = 1:kern.nout,
        %Here expand spring and damper
        for q = 1: kern.nlf,
            % Expand the parameter inverseWidth
            kernLat.inverseWidth =  kern.inverseWidth(q);
            % Compute diag of Kff
            Kff{d,q} = kern.sensitivity(d,q)^2*rbfKernDiagCompute(kernLat, outX{d});
            
            if any(isnan(Kff{d,q})) | any(isinf(Kff{d,q})),
                error('Nan or Inf in Kff')
            end
            
            % Compute Kfu, which corresponds to K_{\hat{f}}u, really.
            Kfu{d,q} = kern.sensitivity(d,q)*rbfKernCompute(kernLat, outX{d}, latX{q});
            
            if any(isnan(Kfu{d,q})) | any(isinf(Kfu{d,q})),
                error('Nan or Inf in Kfu')
            end
        end
    end
end