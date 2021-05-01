function [U]=demo_tv2(data_name)

    movieGen = 0;
    
    %clear;
    close all

    addpath Preprocessing % The preprocessing files
    addpath TV
    tic
    % Method specification -- 1=TV-M,2=TV-m(G.f),3=TV-u,4=TV-u/ME
%     METHOD = 1;
    METHOD = 4;

    % Load Data :
%     data_name = 'face_xy2';

    load(sprintf('%s.mat', data_name))

    %Rearranging face data to be in correct format for this case
%     if strcmp(data_name, 'face_xyn') || strcmp(data_name, 'face_xy') || strcmp(data_name, 'face_xy2') || strcmp(data_name, 'cat_xy') || strcmp(data_name, 'bottle_xy')
%     if strcmp(data_name, 'face_xy') || strcmp(data_name, 'face_xy2') || strcmp(data_name, 'cat_xy') || strcmp(data_name, 'bottle_xy')
    mask = M;
    [m, n, d] = size(I); %Parse inputs
    I = double(reshape(I, m * n, d));
%     mask = double(reshape(mask, m * n, d));
%     end

    % Preprocess the images using the method described in Wu et al 2010 paper
    I = inexact_alm_rpca(I,3/sqrt(length(find(mask>0))), 1e-7,1000);		

    % Solve the UPS problem using TV (see JMIV 2014 paper)
    [N,RHO,L_hat] = ups_tv(I,mask,METHOD);

    %Converting RHO matrix to image
    RHO_8Bit = uint8(255 * mat2gray(RHO));
    [I_Indexed_demotv, map_demotv] = gray2ind(RHO_8Bit, 91);
    save('./PMS_out/I_Indexed_demotv.mat', 'I_Indexed_demotv')
    save('./PMS_out/map_demotv.mat', 'map_demotv')
    
%     if 0
    % Show albedo map
    figure(1); clf
    set(gcf,'color','w');
    imagesc(RHO)
    colormap gray
    axis off
    axis equal
%     end

    % Integrate the normal field and display
    p=reshape(-N(:,2)./(N(:,3)+eps),size(mask));
    q=reshape(N(:,1)./(N(:,3)+eps),size(mask));
    
    U = direct_weighted_poisson(p,q,mask);
    U = U - min(min(U));
    U = U.*mask;
    
    toc

    figure(2); clf
    set(gcf,'color','w');
    surfl(U,[180 90])
    axis equal
    axis off
    shading flat
    colormap gray
    view(130,15)
    %~ camlight headlight

    if 0
        % Project face back onto surface
        if strcmp(data_name, 'face_xy')
            figure(3); clf
            set(gcf,'color','w');
            load('3_cropped.mat')
            load('3_cropped_map.mat')
            warp(1:m, 1:n, U', I_Indexed', map)
            axis off
            axis equal

            figure(4); clf
            set(gcf,'color','w');
            load('3_cropped.mat')
            load('3_cropped_map.mat')
            warp(1:m, 1:n, U', I_Indexed_demotv', map_demotv)
            axis off
            axis equal

        elseif strcmp(data_name, 'face_xy2')
            figure(3); clf
            set(gcf,'color','w');
            load('3_cropped2.mat')
            load('3_cropped_map2.mat')
            warp(1:m, 1:n, U', I_Indexed', map)
            axis off
            axis equal

            figure(4); clf
            set(gcf,'color','w');
            load('3_cropped2.mat')
            load('3_cropped_map2.mat')
            warp(1:m, 1:n, U', I_Indexed_demotv', map_demotv)
            axis off
            axis equal
        end
    end
    
    figure(4); clf
    set(gcf,'color','w');
    warp(1:m, 1:n, U', I_Indexed_demotv', map_demotv)
    axis off
    axis equal
    saveas(gcf, strcat(data_name,'_3Dfig.fig'))
    
    campos([2120, 2000, 2120]);
    saveas(gcf, strcat(data_name,'_3Dfig_view1.png'))
    campos([2120,2000,1000]);
    saveas(gcf, strcat(data_name,'_3Dfig_view2.png'))
    
    if movieGen
        gen_movie(strcat(data_name,'_3Dmov.fig'))
    end

    % Compute calibrated normals
    N_GT=I*pinv(L);
    n_GT=N_GT./repmat(sqrt(sum(N_GT.^2,2)),[1 3]);

    % Compute Mean Angular Error on normals
    inner1=sum(N.*n_GT,2);
    inner2=sum((N*[-1 0 0;0 -1 0; 0 0 1]).*n_GT,2);
    inner3=sum((N*[-1 0 0;0 1 0; 0 0 1]).*n_GT,2);
    inner4=sum((N*[1 0 0;0 -1 0; 0 0 1]).*n_GT,2);
    angle1=acosd(abs(inner1(mask>0)));
    angle2=acosd(abs(inner2(mask>0)));
    angle3=acosd(abs(inner3(mask>0)));
    angle4=acosd(abs(inner4(mask>0)));
    mean_angle1=mean(angle1);
    mean_angle2=mean(angle2);	
    mean_angle3=mean(angle3);	
    mean_angle4=mean(angle4);	
    MAE=min([mean_angle1 mean_angle2 mean_angle3 mean_angle4])
    
    out = 1;
    
    function gen_movie(movie_name)
        mov_counter = 1;
        for x = 2120:-40:1000
          campos([2120, 2000,x])
          drawnow
          Q(mov_counter) = getframe(gcf);
          %pause(0.1)
          mov_counter = mov_counter + 1;
        end
        for x = 2120:-40:-2120
          campos([x,2000,1000])
          drawnow
          Q(mov_counter) = getframe(gcf);
          %pause(0.1)
          mov_counter = mov_counter + 1;
        end
        for x = 1000:40:2120
          campos([-2120, 2000, x])
          drawnow
          Q(mov_counter) = getframe(gcf);
          %pause(0.1)
          mov_counter = mov_counter + 1;
        end
        for x = -2120:40:2120
          campos([x,2000,2120])
          drawnow
          Q(mov_counter) = getframe(gcf);
          %pause(0.1)
          mov_counter = mov_counter + 1;
        end
        %Creating and saving a video using figures stored as frames in Q
        myVideo = VideoWriter(movie_name); %Put appropriate name here to save the movie as
        myVideo.FrameRate = 25; %Specify frame rate of movie
        open(myVideo); %Just a code thing. With videos in Matlab you have to open, write, then close them
        writeVideo(myVideo, Q); %Saving video
        close(myVideo)
    
