function g = gaussianKernGradient_X(kern, x, varargin)

% GAUSSIANKERNGRADIENT Gradient of gaussian kernel's parameters.
% FORMAT
% DESC computes the gradient of
%	functions with respect to the gaussian kernel's
%	parameters. As well as the kernel structure and the input positions,
%	the user provides a matrix PARTIAL which gives the partial
%	derivatives of the function with respect to the relevant elements of
%	the kernel matrix.
% RETURN g:  gradients of the function of interest with respect to the
%	   kernel parameters. The ordering of the vector should match that
%	   provided by the function kernExtractParam.
% ARG kern : the kernel structure for which the gradients are being
%	   computed.
% ARG x : the input locations for which the gradients are being
%	   computed.
% ARG partial : matrix of partial derivatives of the function of
%	   interest with respect to the kernel matrix. The argument takes the
%	   form of a square matrix of dimension  numData, where numData is
%	   the number of rows in X.
%
% FORMAT
% DESC  computes the derivatives
%	as above, but input locations are now provided in two matrices
%	associated with rows and columns of the kernel matrix.
% RETURN g : gradients of the function of interest with respect to the
%	   kernel parameters.
% ARG kern : the kernel structure for which the gradients are being
%	   computed.
% ARG x1 : the input locations associated with the rows of the kernel
%	   matrix.
% ARG x2 : the input locations associated with the columns of the kernel
%	   matrix.
% ARG partial : matrix of partial derivatives of the function of
%	   interest with respect to the kernel matrix. The matrix should have
%	   the same number of rows as X1 and the same number of columns as X2
%	   has rows.
% 
% SEEALSO : gaussianKernParamInit, kernGradient,
% gaussianKernDiagGradient, kernGradX
%  
% COPYRIGHT : Mauricio A. Alvarez and Neil D. Lawrence, 2008
%
% MODIFICATIONS : Cristian Guarnizo, 2015.

% KERN
  
if nargin < 4
    x2 = x;
    covPar = varargin{1};
else
    x2 = varargin{1};
    covPar = varargin{2};
end
%Assuming that each z_m is p-dimensional

[K, ~] = gaussianKernCompute(kern, x, x2);
g = zeros(size(x,1), kern.inputDimension);
temp = (covPar.*K)';
for i = 1:kern.inputDimension,
    matx = repmat(x(:,i),1,size(x,1));
    matx2 = repmat(x2(:,i)',size(x2,1),1);
    g(:,i) = sum(temp.*(matx-matx2));
end
pre2 = 2*kern.precisionU;
g = g(:)'*pre2;