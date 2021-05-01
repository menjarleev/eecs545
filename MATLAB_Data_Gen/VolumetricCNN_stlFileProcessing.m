%Code to specifically consider .stl for volumetric pixel2model outputs

clear; close all; format short g

%% Names of .stl files
output_stl = 'sodapopcoke.stl'; %Name of output .stl file stored in current directory
save_dir = 'VolCNN_PointClouds_Bot101'; %Directory to save point cloud
rot_idx = 0; %Set to 1 to rotate model 90s about z; set to 0 to not

%Specify if to save data
saveData = 1; %Set to 1 to save data, set to 0 to not

%% Loading stl files and converting to point clouds
%Loading output point cloud
output_stl = stlread(fullfile(pwd, output_stl));
output_ptCloud = pointCloud(output_stl.vertices);

%Getting point cloud location data and limits
ptCloud_Locdata = output_ptCloud.Location;
ptCloud_x_lims(1, :) = output_ptCloud.XLimits;
ptCloud_y_lims(1, :) = output_ptCloud.YLimits;
ptCloud_z_lims(1, :) = output_ptCloud.ZLimits;

%Point cloud origins
ptCloud_origins = [mean(ptCloud_x_lims(1, :)),...
                   mean(ptCloud_y_lims(1, :)),...
                   mean(ptCloud_z_lims(1, :))];

if rot_idx
    orig_centered_Locdata = [ptCloud_Locdata(:, 2) - ptCloud_origins(1, 2),...
                           -(ptCloud_Locdata(:, 1) - ptCloud_origins(1, 1)),...
                             ptCloud_Locdata(:, 3) - ptCloud_origins(1, 3)];
else
    orig_centered_Locdata = [ptCloud_Locdata(:, 1) - ptCloud_origins(1, 1),...
                             ptCloud_Locdata(:, 2) - ptCloud_origins(1, 2),...
                             ptCloud_Locdata(:, 3) - ptCloud_origins(1, 3)];
end
                          
%Plotting origin-centered points cloud
if rot_idx
    pcshow(output_ptCloud); view([180 0]); %Setting appropriate viewing angle
else
    pcshow(output_ptCloud); view([90 0]); %Setting appropriate viewing angle
end
cam_pos = get(gca, 'CameraPosition');
close all

if cam_pos(1) > 0
    if rot_idx
        mask = repmat(ptCloud_Locdata(:, 2) > ptCloud_origins(1), 1, 3);
    else
        mask = repmat(ptCloud_Locdata(:, 1) > ptCloud_origins(1), 1, 3);
    end
else
    if rot_idx
        mask = repmat(ptCloud_Locdata(:, 2) < ptCloud_origins(1), 1, 3);
    else
        mask = repmat(ptCloud_Locdata(:, 1) < ptCloud_origins(1), 1, 3);
    end
end

%Removing points which aren't visible based on symmetric of system, and
%scaling same max depth value relative to the camera if 1
selected_points = orig_centered_Locdata.*mask;
orig_centered_Locdata_s = [selected_points(:, 1)/max(abs(selected_points(:, 1))),...
                           selected_points(:, 2),...
                           selected_points(:, 3)];
                            
%Removing points at origin
orig_centered_Locdata_s = orig_centered_Locdata_s(any(orig_centered_Locdata_s, 2), :);
actual_pts = orig_centered_Locdata_s;

%Making saving directory if needed and saving
if saveData
    if ~exist(fullfile(pwd, save_dir), 'dir')
        mkdir(fullfile(pwd, save_dir))
    end
    save(fullfile(pwd, save_dir, 'VolCNN_outputPts.mat'), 'actual_pts');
end