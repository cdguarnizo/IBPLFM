function g = ggXgaussianKernGradient_X(ggKern, gaussianKern, x, x2, covGrad)
% GGXGAUSSIANKERNGRADIENT Compute gradient between the GG and GAUSSIAN kernels.
% FORMAT
% DESC computes the
%	gradient of an objective function with respect to cross kernel terms
%	between GG and GAUSSIAN kernels for the multiple output kernel.
% RETURN g1 : gradient of objective function with respect to kernel
%	   parameters of GG kernel.
% RETURN g2 : gradient of objective function with respect to kernel
%	   parameters of GAUSSIAN kernel.
% ARG ggkern : the kernel structure associated with the GG kernel.
% ARG gaussianKern :  the kernel structure associated with the GAUSSIAN kernel.
% ARG x : inputs for which kernel is to be computed.
%
% FORMAT
% DESC  computes
%	the gradient of an objective function with respect to cross kernel
%	terms between GG and GAUSSIAN kernels for the multiple output kernel.
% RETURN g1 : gradient of objective function with respect to kernel
%	   parameters of GG kernel.
% RETURN g2 : gradient of objective function with respect to kernel
%	   parameters of GAUSSIAN kernel.
% ARG ggKern : the kernel structure associated with the GG kernel.
% ARG gaussianKern : the kernel structure associated with the GAUSSIAN kernel.
% ARG x1 : row inputs for which kernel is to be computed.
% ARG x2 : column inputs for which kernel is to be computed.
%
% SEEALSO : multiKernParamInit, multiKernCompute, ggKernParamInit,
% gaussianKernParamInit
%
% COPYRIGHT : Mauricio A. Alvarez and Neil D. Lawrence, 2008
%
% MODIFICATIONS : Mauricio A. Alvarez, 2009
 
% SHEFFIELDML
 
if nargin < 5
    covGrad = x2;
    x2 = x;
end

g = zeros(size(x2,1), ggKern.inputDimension);

[K, ~, ~, ~, P, ~, ~, ~] = ...
    ggXgaussianKernCompute(ggKern, gaussianKern, x, x2);
temp = covGrad.*K;
for i = 1:gaussianKern.inputDimension,
    matx = repmat(x(:,i),1,size(x2,1));
    matx2 = repmat(x2(:,i)',size(x,1),1);
    g(:,i) = sum(temp.*(matx-matx2))'*P;
end
g = g(:)';