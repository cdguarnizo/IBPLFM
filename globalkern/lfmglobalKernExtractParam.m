function [params, names] = lfmglobalKernExtractParam(kern)

% GGGLOBALKERNEXTRACTPARAM
%
% COPYRIGHT : Mauricio A. Alvarez, 2010
% MODIFICATIONS: C. Guarnizo, 2015.

% MULTIGP

if isfield(kern, 'isVarS') && kern.isVarS,
    if isfield(kern, 'incMass') && kern.incMass,
        params = [kern.inverseWidth(:)' kern.mass(:)' kern.spring(:)' kern.damper(:)'];
    else
        params = [kern.inverseWidth(:)' kern.spring(:)' kern.damper(:)'];
    end
    if nargout > 1,
        invWidthNames = cell(kern.nlf);
        for i=1:kern.nlf
            force = num2str(i);
            invWidthNames{i} = ['inverse width force ' force '.'];
        end
        
        springNames = cell(kern.nout);
        for i=1:kern.nout,
            output = num2str(i);
            springNames{i} = ['spring output ' output '.'];
        end
        
        damperNames = cell(kern.nout);
        for i=1:kern.nout,
            output = num2str(i);
            damperNames{i} = ['damper output ' output '.'];
        end

        massNames = cell(kern.nout);
        if isfield(kern, 'incMass') && kern.incMass,
            for i=1:kern.nout,
                output = num2str(i);
                massNames{i} = ['mass output ' output '.'];
            end
        end
        
        names = [invWidthNames(:)' massNames(:)' springNames(:)'... 
            damperNames(:)'];
    end
else
    if isfield(kern, 'incMass') && kern.incMass,
        params = [kern.inverseWidth(:)' kern.mass(:)' kern.spring(:)' kern.damper(:)' kern.sensitivity(:)'];
    else
        params = [kern.inverseWidth(:)' kern.spring(:)' kern.damper(:)' kern.sensitivity(:)'];
    end
    
    if nargout > 1,
        invWidthNames = cell(kern.nlf);
        for i=1:kern.nlf
            force = num2str(i);
            invWidthNames{i} = ['inverse width force ' force '.'];
        end
        
        springNames = cell(kern.nout);
        for i=1:kern.nout,
            output = num2str(i);
            springNames{i} = ['spring output ' output '.'];
        end
        
        damperNames = cell(kern.nout);
        for i=1:kern.nout,
            output = num2str(i);
            damperNames{i} = ['damper output ' output '.'];
        end

        massNames = cell(kern.nout);
        if isfield(kern, 'incMass') && kern.incMass,
            for i=1:kern.nout,
                output = num2str(i);
                massNames{i} = ['mass output ' output '.'];
            end
        end

        sensitivityNames = cell(kern.nout, kern.nlf);
        for i=1:kern.nout,
            output = num2str(i);
            for j=1:kern.nlf,
                force = num2str(j);
                sensitivityNames{i,j} = ['sensitivity output ' output ' force ' force '.'];
            end
        end
        
        names = [invWidthNames(:)' massNames(:)' springNames(:)'... 
            damperNames(:)' sensitivityNames(:)'];
    end
end