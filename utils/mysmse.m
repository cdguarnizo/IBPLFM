function smse = mysmse2(ytrue,ypred)
%MYSMSE smse = mysmse(ytrue, ypred,trainmean)
%   Compute the standardised mean square error (SMSE), also NMSE in some
%   publications.
% 
% SMSE = mean square error / mean test variance
smse=mean((ypred-ytrue).^2)/mean((mean(ytrue)-ytrue).^2);
end