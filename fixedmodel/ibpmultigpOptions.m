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
% COPYRIGHT : Neil D. Lawrence, Mauricio Alvarez, 2008
%
% MODIFICATIONS : Cristian Guarnizo, 2014

% IBPMULTIGP

options = multigpOptions(approx);
options.type = 'ibpmultigp';

options.sparsePriorType = 'ibp';
options.optimiser = 'scg';

options.isVarS = false; %If ARD or SpikeSlab this should be true
options.gammaPrior = false;
options.InitSearchS = false;

options.fixinducing = true;
options.Trainkern = true;
options.InitKern = true;
options.debug = false;

options.sorteta = true; %If ARD then this should be false
options.isVarU = true;
options.OptMarU = true;
options.IBPisInfinite = true;
options.Opteta = false;
options.force_posUpdate = false;
options.UseMeanConstants = false;