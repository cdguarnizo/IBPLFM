function options = ibpmultigpOptions(approx)

% IBPMULTIGPOPTIONS Return default options for the IBP LFM model.
% FORMAT
% DESC returns the default options in a structure for a MULTIGP model.
% ARG approx : approximation type, either 'none' (no approximation),
% 'fitc' (fully
% independent training conditional) or 'pitc' (partially
% independent training conditional.
% RETURN options : structure containing the default options for the
% given approximation type.
%
% SEEALSO : multigpCreate
%
% COPYRIGHT : Mauricio Alvarez, 2008
%
% COPYRIGHT : Neil D. Lawrence, 2008

% IBPMULTIGP

options = multigpOptions(approx);
options.type = 'ibpmultigp';

