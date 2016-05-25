function [Xf, yf, y, X, gpModel, SNRlog, index] = selectChannelsMocap(subject, ...
    sample, P, threshold)

% FORMAT
% DESC returns selected channels from a mocap example. Selection is based
% on SNR
% RETURN Xf : inputs selected channels
% RETURN yf : outputs selected channels
% RETURN X : inputs decimated original channels
% RETURN y : outputs decimated original channels
% RETURN gpModel : trained single-output GPs
% RETURN SNRlog : SNR for all the output channels
% RETURN index : index of the selected channels
% ARG subject : number of mocap subject
% ARG sample : number of exercise performed by the subject
% ARG P : decimation factor
% ARG threshold : value of 10log10(SNR) for selecting channels
%
% COPYRIGHT : Mauricio A. Alvarez, 2014

if exist(['snr_' num2str(subject) '_' num2str(sample) '.mat'], 'file')
    load(['snr_' num2str(subject) '_' num2str(sample) '.mat']);
else    
    %%%%%%%%%%% READ THE MOCAP %%%%%%%%%%%%%%%%%%
    % Add the MOCAP toolbox
    %addpath(genpath('../../mlprojects2/mocap/'))
    %addpath(genpath('../../mlprojects2/ndlutil/'))
    % Parameters of the simulation
    Nf = 120;               % Number of fps in the original data
    % Reading the skeleton
    skel = acclaimReadSkel([num2str(subject) '.asf']);
    saveSkeleton = false;
    if saveSkeleton
        fid = fopen(strcat(int2str(subject),'_skeleton.txt'),'w');
        fid2 = fopen(strcat(int2str(subject),'_channels.txt'),'w');
        cont = 1;
        for i=1:length(skel.tree)
            part_name = skel.tree(i).name;
            fprintf(fid,'%s',part_name);
            for k=1:length(skel.tree(i).channels)
                channel_name = skel.tree(i).channels{k};
                fprintf(fid,' %s',channel_name);
                fprintf(fid2,'Channel %2g : %s - %s',cont,part_name,channel_name);
                fprintf(fid2,'\n');
                cont = cont+1;
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
        fclose(fid2);
    end
    % Loading the movement data
    if sample < 10
        channels = acclaimLoadChannels(['./' num2str(subject) ...
            '_0' num2str(sample) '.amc'], skel);
    else
        channels = acclaimLoadChannels(['./' num2str(subject) ...
            '_' num2str(sample) '.amc'], skel);
    end
    [nx, np] = size(channels);
    nxd = ceil(nx/P);
    tx = ((1:nxd)*(P/Nf))';
    % Decimation of original channels by P
    decimatedChannels = zeros(nxd, np);
    for i=1:np
        decimatedChannels(:,i) = decimate(channels(:,i), P);
    end
    
    
    %%%%%%%%%%% Train GPs and compute SNR based on the variances %%%%%%%%%%%%%
    X = cell(1, np);
    y = cell(1, np);
    for i = 1:np
        y{i} = decimatedChannels(:, i);
        X{i} = tx;
    end
    
    %addpath(genpath('../../mlprojects2/gp/'));
    %addpath(genpath('../../mlprojects2/optimi/'));
    %addpath(genpath('../../mlprojects2/kern/'));
    %addpath(genpath('../../mlprojects2/matlab/netlab/'));
    
    options = gpOptions('ftc');
    options.kern = {'rbf', 'white'};
    options.scale2var1 = 1;
    itersSingleGp = 500;
    % Configuration of parameters
    gpModel = cell(length(y), 1);
    signal2noise = zeros(length(y), 1);
    params = zeros(1,2*length(y));
    for j=1:length(y)
        gpModel{j} = gpCreate(1, 1, X{j}, y{j}, options);
        gpModel{j} = gpOptimise(gpModel{j}, 1, itersSingleGp);
        %paramt = modelExtractParam(gpModel{j});
        %params([j length(y)+j]) = paramt([2 3]);
        signal2noise(j) = gpModel{j}.kern.comp{1}.variance/gpModel{j}.kern.comp{2}.variance;
    end
    
    %%%%%%%%%%% Set a threshold and select the number of channels %%%%%%%%%%%%
    
    SNRlog = 10*log10(signal2noise);
    index = find(SNRlog > threshold);
    yf = y(SNRlog > threshold);
    Xf = X(SNRlog > threshold);
    
    save(['snr_' num2str(subject) '_' num2str(sample) '.mat'], ...
        'yf', 'Xf', 'y', 'X', 'gpModel', 'SNRlog', 'index')
    
    
    rmpath(genpath('../../mlprojects2/mocap/'))
    rmpath(genpath('../../mlprojects2/ndlutil/'))
    rmpath(genpath('../../mlprojects2/gp/'));
    rmpath(genpath('../../mlprojects2/optimi/'));
    rmpath(genpath('../../mlprojects2/kern/'));
    rmpath(genpath('../../mlprojects2/matlab/netlab/'));    
end





