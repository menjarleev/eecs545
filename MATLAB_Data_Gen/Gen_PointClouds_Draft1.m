%% Code to convert all .stl files to point cloud files and format for validation

clear; format short g; close all;

%% User Inputs
%Flag to specify whether to save data or not
saveData = 1;

%Indices of .stl files to get point cloud data for
bottle_idx = 1:105;

%Indices to rotate as needed
rot_idx = 18;

%Directory where STL_Files are stored
main_dir = 'STL_Files';

%Directory to save files
save_dir = 'Actual_Bottle_Pts';

%% No Modifications Past This Point
%Loading all .stl files
raw_stl = cell(length(bottle_idx), 1);
ptClouds = cell(length(bottle_idx), 1);
ptCloud_Locdata = cell(length(bottle_idx), 1);
cam_pos = cell(length(bottle_idx), 1);
orig_centered_Locdata = cell(length(bottle_idx), 1);
orig_centered_Locdata_s = cell(length(bottle_idx), 1);
ptCloud_x_lims  = zeros(length(bottle_idx), 2);
ptCloud_y_lims  = zeros(length(bottle_idx), 2);
ptCloud_z_lims  = zeros(length(bottle_idx), 2);
ptCloud_origins = zeros(length(bottle_idx), 3);
for img = 1:length(bottle_idx)
    %Loading stl file
    raw_stl{img} = stlread(fullfile(pwd, main_dir, sprintf('bottle_%d.stl', bottle_idx(img))));
    
    %Changing to point-cloud format
    ptClouds{img} = pointCloud(raw_stl{img}.vertices);
    
    %Getting point cloud location data and limits
    ptCloud_Locdata{img} = ptClouds{img}.Location;
    ptCloud_x_lims(img, :) = ptClouds{img}.XLimits;
    ptCloud_y_lims(img, :) = ptClouds{img}.YLimits;
    ptCloud_z_lims(img, :) = ptClouds{img}.ZLimits;
    
    %Point cloud origins
    ptCloud_origins(img, :) = [mean(ptCloud_x_lims(img, :)),...
                               mean(ptCloud_y_lims(img, :)),...
                               mean(ptCloud_z_lims(img, :))];
                           
                           
    %Finding points which are closet to the camera
    if bottle_idx(img) == rot_idx
        orig_centered_Locdata{img} = [ptCloud_Locdata{img}(:, 2) - ptCloud_origins(img, 2),...
                                      -(ptCloud_Locdata{img}(:, 1) - ptCloud_origins(img, 1)),...
                                      ptCloud_Locdata{img}(:, 3) - ptCloud_origins(img, 3)];
    else
        orig_centered_Locdata{img} = [ptCloud_Locdata{img}(:, 1) - ptCloud_origins(img, 1),...
                                      ptCloud_Locdata{img}(:, 2) - ptCloud_origins(img, 2),...
                                      ptCloud_Locdata{img}(:, 3) - ptCloud_origins(img, 3)];
    end
                         
    %Plotting origin-centered points cloud
    if bottle_idx(img) == rot_idx
        pcshow(ptClouds{img}); view([180 0]); %Setting appropriate viewing angle
    else
        pcshow(ptClouds{img}); view([90 0]); %Setting appropriate viewing angle
    end
    cam_pos{img} = get(gca, 'CameraPosition');
    close all
    
    if cam_pos{img}(1) > 0
        if bottle_idx(img) == rot_idx
            mask = repmat(ptCloud_Locdata{img}(:, 2) > ptCloud_origins(img, 1), 1, 3);
        else
            mask = repmat(ptCloud_Locdata{img}(:, 1) > ptCloud_origins(img, 1), 1, 3);
        end
    else
        if bottle_idx(img) == rot_idx
            mask = repmat(ptCloud_Locdata{img}(:, 2) < ptCloud_origins(img, 1), 1, 3);
        else
            mask = repmat(ptCloud_Locdata{img}(:, 1) < ptCloud_origins(img, 1), 1, 3);
        end
    end
    
    %Removing points which aren't visible based on symmetric of system, and
    %scaling same max depth value relative to the camera if 1
    selected_points = orig_centered_Locdata{img}.*mask;
    orig_centered_Locdata_s{img} = [selected_points(:, 1)/max(abs(selected_points(:, 1))),...
                                    selected_points(:, 2),...
                                    selected_points(:, 3)];

    %Removing points at origin
    orig_centered_Locdata_s{img} = orig_centered_Locdata_s{img}(any(orig_centered_Locdata_s{img}, 2), :);
                                
    %Saving data
    if saveData
        if ~exist(fullfile(pwd, save_dir), 'dir')
            mkdir(fullfile(pwd, save_dir))
        end
        actual_pts = orig_centered_Locdata_s{img};
        save(fullfile(pwd, save_dir, sprintf('bottle_%d_actualPts.mat', bottle_idx(img))), 'actual_pts')
    end
%     
%     %Plotting
%     figure(1); clf
%     scatter3(orig_centered_Locdata_s{img}(:, 1),...
%              orig_centered_Locdata_s{img}(:, 2),...
%              orig_centered_Locdata_s{img}(:, 3));
%     view([125 30]);
                                
    fprintf('%d/%d Bottles Completed!\n', img, length(bottle_idx))
end