function y = logitTransform(x, transform)

% EXPTRANSFORM Constrains a parameter to be between 0 and 1.
% FORMAT
% DESC contains commands to constrain parameters to be positive via
% exponentiation.
% ARG x : input argument.
% ARG y : return argument.
% ARG transform : type of transform, 'atox' maps a value into
% the transformed space (i.e. makes it positive). 'xtoa' maps the
% parameter back from transformed space to the original
% space. 'gradfact' gives the factor needed to correct gradients
% with respect to the transformed parameter.
% 
% SEEALSO : negLogLogitTransform, sigmoidTransform
%
% COPYRIGHT : Cristian Guarnizo, 2016

% SHEFFIELDML


limVal = 36;
y = zeros(size(x));
switch transform
 case 'atox'
  index = find(x<-limVal);
  y(index) = 1./(1+exp(limVal));
  x(index) = NaN;
  index = find(x>limVal);
  y(index) = 1./(1+exp(-limVal));
  x(index) = NaN;
  index = find(~isnan(x));
  if ~isempty(index),
    y(index) = 1./(1+exp(-x(index)));
  end
 case 'xtoa'
  y = log(x+(x==0.)) - log(1 - x + (x==1.));
 case 'gradfact'
  y = x.*(1-x);
end
