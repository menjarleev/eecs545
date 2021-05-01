%Code to creating data samples for each bottle for each lighting condition

clear; format short g; close all

%% User Inputs
%Indices of reference lighting condition images and face images to consider
light_images  = 2:36; %Indices of all relit images
bottle_images = 1:105; %Indices of all bottle data set
input_image   = 2; %Index of image to use for input

%Indices of samples for training, validation, and test
train_ind = 1:95;  %Training
val_ind   = 96:100; %Validation
test_ind  = 101:105; %Test

%Desired image size
im_size = [128, 128];

%% No Modifications Past This Point
%Number of reference lighting condition images and face images to consider
num_light_images = length(light_images);
num_bottle_images = length(bottle_images);

%Strings for original, x-flip, and y-flip
str_orient = {'orig', 'yflip'};

%Strings for rotations
str_rot = {'0', '90', '180', '270'};

%Strings for translations
str_trans = {'0', '_x1_n', '_x1_p', '_x2_n', '_x2_p',...
                  '_y1_n', '_y1_p', '_y2_n', '_y2_p'};
         

%Cells to store flipped and rotations (no translations) 
c_ims_fr_input  = cell(num_bottle_images, length(str_orient), length(str_rot));
c_ims_fr_output = cell(num_bottle_images, length(str_orient), length(str_rot), num_light_images-1);

%Cells to store translations
c_ims_t_input  = cell(num_bottle_images, length(str_orient), (length(str_trans)-1));
c_ims_t_output = cell(num_bottle_images, length(str_orient), (length(str_trans)-1), num_light_images-1);
              
%Matrices to store all images
c_ims_mat_input = zeros(num_bottle_images,...
                       (length(str_orient)*length(str_rot)) + (length(str_orient)*(length(str_trans)-1)),...
                       im_size(1), im_size(2));
c_ims_mat_output = zeros(num_bottle_images, num_light_images-1,...
                        (length(str_orient)*length(str_rot)) + (length(str_orient)*(length(str_trans)-1)),...
                        im_size(1), im_size(2));

%Loading all images for each case
main_dir = 'Bottle_Images';

for jj = 1:num_bottle_images
    counter_1 = 1; %Counter for rotation and flipping only
    counter_2 = 1; %Counter for translation and flipping only
    input_flag = 0; %Flag for indexing when saving output

    %Folder name to store data
    if any(jj == train_ind)
        store_folder = sprintf('Training_Data_%d', im_size(1));
    elseif any(jj == val_ind)
        store_folder = sprintf('Validation_Data_%d', im_size(1));
    else
        store_folder = sprintf('Test_Data_%d', im_size(1));
    end
    
    %String for alphabetical file naming
    if jj < 10
        app_str_dir = '00';
    elseif jj < 100 && jj > 9
        app_str_dir = '0';
    elseif jj > 99
        app_str_dir = '';
    end
    
    counter_4 = 1; %Counter for storing index of relit images
    for ii = 1:num_light_images
        counter_3 = 1; %Counter of storing entire data
        
        %Current directory
        c_dir = sprintf('Bottle_%d', bottle_images(jj));
        
        %Looping through cases of rotations and flips (no translations)
        for o_loop = 1:length(str_orient)
            for r_loop = 1:length(str_rot)
                if light_images(ii) == input_image
                    c_ims_fr_input{jj, o_loop, r_loop} = imresize(im2double(imread(fullfile(pwd, main_dir, c_dir,...
                                                         sprintf('%d_%s_rot%s_trans%s.png',...
                                                         light_images(ii), str_orient{o_loop},...
                                                         str_rot{r_loop}, str_trans{1})))) * 255, im_size);
                    c_ims_mat_input(jj, counter_3, :, :) = c_ims_fr_input{jj, o_loop, r_loop};
                else
                    c_ims_fr_output{jj, o_loop, r_loop, counter_1} = imresize(im2double(imread(fullfile(pwd, main_dir, c_dir,...
                                                         sprintf('%d_%s_rot%s_trans%s.png',...
                                                         light_images(ii), str_orient{o_loop},...
                                                         str_rot{r_loop}, str_trans{1})))) * 255, im_size);  
                    c_ims_mat_output(jj, counter_4, counter_3, :, :) = c_ims_fr_output{jj, o_loop, r_loop, counter_1};
                end
                counter_3 = counter_3 + 1;
            end
        end
        if ~(light_images(ii) == input_image)
            counter_1 = counter_1 + 1; 
        end
        
        %Looping through translation cases
        for o_loop = 1:length(str_orient)
            for t_loop = 2:length(str_trans)
                if light_images(ii) == input_image
                    c_ims_t_input{jj, o_loop, t_loop-1} = imresize(im2double(imread(fullfile(pwd, main_dir, c_dir,...
                                                        sprintf('%d_%s_rot%s_trans%s.png',...
                                                        light_images(ii), str_orient{o_loop},...
                                                        str_rot{1}, str_trans{t_loop})))) * 255, im_size);
                    c_ims_mat_input(jj, counter_3, :, :) = c_ims_t_input{jj, o_loop, t_loop-1};
                else
                    c_ims_t_output{jj, o_loop, t_loop-1, counter_2} = imresize(im2double(imread(fullfile(pwd, main_dir, c_dir,...
                                                        sprintf('%d_%s_rot%s_trans%s.png',...
                                                        light_images(ii), str_orient{o_loop},...
                                                        str_rot{1}, str_trans{t_loop})))) * 255, im_size);
                    c_ims_mat_output(jj, counter_4, counter_3, :, :) = c_ims_t_output{jj, o_loop, t_loop-1, counter_2};
                end
                counter_3 = counter_3 + 1;
            end
        end
        if ~(light_images(ii) == input_image)
            counter_2 = counter_2 + 1; 
        end
        
        %String for alphabetical file naming
        if (ii < 10 && input_flag == 0) || (ii < 11 && input_flag == 1)
            app_str_out= '00';
        elseif (ii < 100 && ii > 9 && input_flag == 0) || (ii < 101 && ii > 10 && input_flag == 1)
            app_str_out = '0';
        elseif (ii > 99 && input_flag == 0) || (ii > 100 && input_flag == 1) 
            app_str_out = '';
        end
        
        %Saving input and output matrices
        if ~exist(fullfile(pwd, store_folder, sprintf('Bottle_%s%d', app_str_dir, jj)), 'dir')
            mkdir(fullfile(pwd, store_folder, sprintf('Bottle_%s%d', app_str_dir, jj)))
        end
        if light_images(ii) == input_image
            input = squeeze(c_ims_mat_input(jj, :, :, :));
            save(fullfile(pwd, store_folder, sprintf('Bottle_%s%d', app_str_dir, jj), 'input.mat'), 'input')
            input_flag = 1;
        else
            output = squeeze(c_ims_mat_output(jj, counter_4, :, :, :));
            save(fullfile(pwd, store_folder, sprintf('Bottle_%s%d', app_str_dir, jj), sprintf('output_%s%d.mat', app_str_out, light_images(ii) - min(light_images) + 1 - input_flag)), 'output')
        end
        if ~(light_images(ii) == input_image)
            counter_4 = counter_4 + 1;  
        end
    end
    fprintf('%d/%d bottle sets completed!\n', jj, num_bottle_images)
end

%Saving training data
input  = c_ims_mat_input(train_ind, :, :, :);
save(fullfile(pwd, sprintf('Training_Data_%d', im_size(1)), 'input'), 'input')
for ii = 1:(num_light_images-1)
    output = squeeze(c_ims_mat_output(train_ind, ii, :, :, :));
    save(fullfile(pwd, sprintf('Training_Data_%d', im_size(1)), sprintf('output_%d.mat', light_images(ii) - min(light_images) + 1)), 'output')
end
fprintf('Training Data Saved!\n')

%Saving validation data
input  = c_ims_mat_input(val_ind, :, :, :);
save(fullfile(pwd, sprintf('Validation_Data_%d', im_size(1)), 'input'), 'input')
for ii = 1:(num_light_images-1)
    output = squeeze(c_ims_mat_output(val_ind, ii, :, :, :));
    save(fullfile(pwd, sprintf('Validation_Data_%d', im_size(1)), sprintf('output_%d.mat', light_images(ii) - min(light_images) + 1)), 'output')
end
fprintf('Validation Data Saved!\n')

%Saving test data
input  = c_ims_mat_input(test_ind, :, :, :);
save(fullfile(pwd, sprintf('Test_Data_%d', im_size(1)), 'input'), 'input')
for ii = 1:(num_light_images-1)
    output = squeeze(c_ims_mat_output(test_ind, ii, :, :, :));
    save(fullfile(pwd, sprintf('Test_Data_%d', im_size(1)), sprintf('output_%d.mat', light_images(ii) - min(light_images) + 1)), 'output')
end
fprintf('Test Data Saved!\n')