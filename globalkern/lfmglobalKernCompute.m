function [Kff, Kfu, Kuu] = lfmglobalKernCompute(kern, outX, latX, gamma)

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
                    Kfft{d,dp} = real(kern.funcNames.computeCross(kernOut, kernOut2, outX{d}, outX{dp}));
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
    kernOut = kern.template.output;
    
    % Compute Kuu -> rbf kernel
    for k = 1:kern.nlf,
        % First we need to expand the parameters in the vector to the local
        % kernel
        kernLat.inverseWidth = kern.inverseWidth(k);
        Kuu{k} = real(kern.funcNames.computeLat(kernLat, latX{k}));
        if ~isempty(gamma) %Ask Mauricio about this conditional
            Kuu{k} = Kuu{k} + gamma(k)*eye(size(Kuu{k}));
        end
    end
    for i = 1:kern.nout,
        %Here expand spring and damper
        kernOut.mass = kern.mass(i);
        kernOut.damper = kern.damper(i);
        kernOut.spring = kern.spring(i);
        for j = 1: kern.nlf,
            kernOut.sensitivity = kern.sensitivity(i,j);
            % Expand the parameter inverseWidth
            kernOut.inverseWidth = kern.inverseWidth(j);
            kernLat.inverseWidth =  kern.inverseWidth(j);
            % Compute Kff
            Kff{i,j} = real(kern.funcNames.computeOut(kernOut, outX{i}));
            
            if any(isnan(Kff{i,j})),
                error('Nan in Kff')
            end
            
            % Compute Kfu, which corresponds to K_{\hat{f}}u, really.
            Kfu{i,j} = real(kern.funcNames.computeCross(kernOut, kernLat, outX{i}, latX{j}));
        end
    end
end