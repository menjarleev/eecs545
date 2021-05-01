%Generating all lighting conditions based on provided .stl files to match
%output from model

clear; format short g

%% User Inputs
%Flag to store all data
save_data = 1;

%Indices of .stl files to make data for
bottle_idx = 101:105;

%Output figure size
out_fig_size = [128, 128];

%Index of studio lighting condition in L
s_idx = 2;

%Indices of output lighting conditions
out_idx = 3:36;

%Indices of samples for training, validation, and test
train_ind = 1:95;  %Training
val_ind   = 96:100; %Validation
test_ind  = 101:105; %Test

%Additionally suffix for loading a specific set of augmented images
aug_img_str = '';

%% No Modifications Past This Point
%Directory to load images
main_dir_load = 'Bottle_Images';
 
img_out_counter = 1;
prog_counter = 1;
for img = bottle_idx
    %Directory to store output images
    if any(img == train_ind)
        main_dir_out = sprintf('Training_Data_%d_ActModData_Output%s', out_fig_size(1), aug_img_str);
    elseif any(img == val_ind)
        main_dir_out = sprintf('Validation_Data_%d_ActModData_Output%s', out_fig_size(1), aug_img_str);
    else
        main_dir_out = sprintf('Test_Data_%d_ActModData_Output%s', out_fig_size(1), aug_img_str);
    end
    if ~exist(main_dir_out, 'dir')
        mkdir(main_dir_out)
    end

    %Directory to store studio image
    if any(img == train_ind)
        main_dir_stud = sprintf('Training_Data_%d_ActModData_Studio%s', out_fig_size(1), aug_img_str);
    elseif any(img == val_ind)
        main_dir_stud = sprintf('Validation_Data_%d_ActModData_Studio%s', out_fig_size(1), aug_img_str);
    else
        main_dir_stud = sprintf('Test_Data_%d_ActModData_Studio%s', out_fig_size(1), aug_img_str);
    end
    if ~exist(main_dir_stud, 'dir')
        mkdir(main_dir_stud)
    end
    
    %Defining directory to store images for specific bottles
    if ~exist(fullfile(pwd, main_dir_out), 'dir')
        mkdir(fullfile(pwd, main_dir_out))
    end
    %Defining directory to store studio lighting condition images
    if ~exist(fullfile(pwd, main_dir_stud), 'dir')
        mkdir(fullfile(pwd, main_dir_stud))
    end
    
    %Current directory
    c_img_dir = sprintf('Bottle_%d', img);
    
    %Loading studio images
    stud_img = imread(fullfile(pwd, main_dir_load, c_img_dir, sprintf('%d%s.png', s_idx, aug_img_str)));
    
    %Saving studio images to new folder
    imwrite(imresize(stud_img, out_fig_size), ...
            fullfile(pwd, main_dir_stud, sprintf('%d%s.jpg', img, aug_img_str))); %Save as .jpg file
        
    %Loading and rewriting output images
    for i_out = out_idx
        %Loading output images
        out_img = imread(fullfile(pwd, main_dir_load, c_img_dir, sprintf('%d.png', i_out)));
    
        %Saving output images to new folder
        imwrite(imresize(out_img, out_fig_size), ...
                fullfile(pwd, main_dir_out, sprintf('%d.jpg', img_out_counter))); %Save as .jpg file
        
        %Updating counter
        img_out_counter = img_out_counter + 1;
    end
    
    %Printout of progress and updating progress counter
    fprintf('Bottle %d completed! (%d/%d)\n', img, prog_counter, length(bottle_idx))
    prog_counter = prog_counter + 1;
    
end