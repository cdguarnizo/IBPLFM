function [params, names] = icmglobalKernExtractParam(kern)

% GGGLOBALKERNEXTRACTPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010
% MODIFICATIONS: C. Guarnizo, 2017.

% MULTIGP

params = [kern.inverseWidth(:)' kern.sensitivity(:)'];

if nargout > 1,
    invWidthNames = cell(kern.nlf);
    for i = 1:kern.nlf,
        force = num2str(i);
        invWidthNames{i} = ['inverse width latent function ' force '.'];
    end
    
    sensitivityNames = cell(kern.nout, kern.nlf);
    for i=1:kern.nout,
        output = num2str(i);
        for j=1:kern.nlf,
            force = num2str(j);
            sensitivityNames{i,j} = ['sensitivity output ' output ' force ' force '.'];
        end
    end
    
    names = [invWidthNames(:)' sensitivityNames(:)'];
end
