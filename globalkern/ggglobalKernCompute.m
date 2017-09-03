function [Kff, Kfu, Kuu] = ggglobalKernCompute(kern, outX, latX, gamma)

% GGGLOBALKERNCOMPUTE
% COPYRIGTH : Mauricio A. Alvarez, 2013, Cristian Guarnizo, 2014, 2015.
% MULTIGP

if nargin < 4
    gamma = [];
end

if strcmp(kern.approx,'ftc'),
    Kfft = cell(kern.nout, kern.nout);
    Kff = cell(kern.nlf,1);
    kernOut = kern.template.output;
    kernOut2 = kern.template.output;
    for q = 1:kern.nlf,
        kernOut.precisionU = kern.precisionU(q);
        kernOut2.precisionU = kern.precisionU(q);
        for d = 1:kern.nout,
            kernOut.precisionG = kern.precisionG(d);
            kernOut.sensitivity = kern.sensitivity(d,q);
            Kfft{d,d} = real(kern.funcNames.computeOut(kernOut, outX{d}));
            for dp = d+1:kern.nout,
                kernOut2.precisionG = kern.precisionG(dp);
                kernOut2.sensitivity = kern.sensitivity(dp,q);
                Kfft{d,dp} = real(kern.funcNames.computeCrossOut(kernOut, kernOut2, outX{d}, outX{dp}));
                Kfft{dp,d} = Kfft{d,dp}.';
            end
        end
        Kff{q} = cell2mat(Kfft);
        %Kff = Kff + Kfu{q};
    end
    
else
    if nargin > 1,
        Kff = cell(kern.nout, kern.nlf);
        kernOut = kern.template.output;
    end
    if nargin > 2,
        flagu = 1;
        Kuu = cell(kern.nlf,1);
        Kfu = cell(kern.nout, kern.nlf);
        kernLat = kern.template.latent;
    end
    if flagu,
        % Compute Kuu
        for k = 1:kern.nlf,
            % First we need to expand the parameters in the vector to the local
            % kernel
            kernLat.precisionU = kern.precisionU(k);
            Kuu{k} = real(kern.funcNames.computeLat(kernLat, latX{k}));
            if ~isempty(gamma),
                Kuu{k} = Kuu{k} + gamma(k)*eye(size(Kuu{k}));
            end
        end
    end
    for d = 1:kern.nout,
        % Expand the parameter decay
        kernOut.precisionG = kern.precisionG(d);
        for q = 1:kern.nlf,
            kernOut.sensitivity = kern.sensitivity(d,q);
            % Expand the parameter inverseWidth
            kernOut.precisionU = kern.precisionU(q);
            kernLat.precisionU = kern.precisionU(q);
            % Compute Kff
            Kff{d,q} = real(kern.funcNames.computeOut(kernOut, outX{d}));
            
            if any(isnan(Kff{d,q})) | any(isinf(Kff{d,q})),
                error('Nan or Inf in Kff')
            end
            % Compute Kfu, which corresponds to K_{\hat{fu}}, really.
            Kfu{d,q} = real(kern.funcNames.computeCross(kernOut, kernLat, outX{d}, latX{q}));
            
            if any(isnan(Kfu{d,q})) | any(isinf(Kfu{d,q})),
                error('Nan or Inf in Kfu')
            end
        end
    end
end