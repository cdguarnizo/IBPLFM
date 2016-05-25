function [params, names] = simglobalKernExtractParam(kern)

% GGGLOBALKERNEXTRACTPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010

% MULTIGP

if isfield(kern, 'isVarS') && kern.isVarS,
    params = [kern.inverseWidth(:)' kern.decay(:)'];
    if nargout > 1,
        invWidthLatNames = cell(1, kern.nlf);
        for i=1:kern.nlf
            force = num2str(i);
            invWidthLatNames{i} = ['inverse width latent: force ' force '.'];
        end

        invWidthOutNames = cell(1, kern.nout);
        for i=1:kern.nout
            output = num2str(i);
            invWidthOutNames{i} = ['decay output: output ' output '.'];
        end

        names = [invWidthLatNames(:)' invWidthOutNames(:)'];
    end
else
    params = [kern.inverseWidth(:)' kern.decay(:)' kern.sensitivity(:)'];
    if nargout > 1,
        invWidthLatNames = cell(1, kern.nlf);
        for i=1:kern.nlf,
            force = num2str(i);
            invWidthLatNames{i} = ['inverse width latent: force ' force '.'];
        end

        invWidthOutNames = cell(1, kern.nout);
        for i=1:kern.nout,
            output = num2str(i);
            invWidthOutNames{i} = ['decay output: output ' output '.'];
        end
    
        for i=1:kern.nout,
            output = num2str(i);
            for j=1:kern.nlf,
                force = num2str(j);
                sensitivityNames{i,j} = ['sensitivity output ' output ' force ' force '.'];
            end
        end
        names = [invWidthLatNames(:)' invWidthOutNames(:)' sensitivityNames(:)'];
    end
end