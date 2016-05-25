close all
addpath('lfm2');
% fd = amc_to_matrix('35_17.amc');
% Inddel = [1 3 7 10 13 16 18 19 22 24 25 26 31 32 33 34 35 36 37 38 43:48 54 55 62];
% fd(:, Inddel) = [];
% 
% for k=1:size(fd,2),
%     figure(k);
%     plot(fd(:,k));
% end

fd = amc_to_matrix('49_18.amc');
IndSel = [1 2:10 13 16:23 27:32 35 39:41 43 49:61];
%IndSel = [5 6 8 9 11 14:20 22 23 27:31 39:43 49 51:53 55 56 58 59];
fd = fd(:,IndSel);

for k=1:size(fd,2),
    figure(IndSel(k));
    plot(fd(:,k));
end