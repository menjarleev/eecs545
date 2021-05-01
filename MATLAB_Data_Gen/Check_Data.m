%% Code to check if data was prepared correctly

clear; format short g; close all;

%% User Inputs
%----------Parameters for checking input-output data----------%
%Flag to create and/or save generated images
make_check_fig = 0; %Set to 1 to create fig; 0 to not
save_fig = 1; %Set to 1 to save created fig; 0 to not

%Indices of bottles images to consider for check figures
output_idx = 1:34; %Indices of output images to load
bottle_idx = 11:105; %Indices of all bottle data set
im_idx = 1:24; %Indices of images within input and output to consider

%Size of figure to check images
fig_format = [5, 7];

%----------Parameters for creating figure of all augmentions----------%
make_aug_fig = 1; %Set to 1 to make figure; 0 to not
save_aug_fig = 1; %Set to 1 to save augmented fig; 0 to not

%Indices of bottle images to consider for augmentation view figures
bottle_idx_aug = 1;
im_idx_aug = 1:24; %Indices of augmentations to consider

aug_fig_idx = 0; %Set to 0 to use input image; set to output index for specific other lighting condition
aug_fig_format = [6, 4];

%----------Parameters needed and are the same for both figures----------%
%Indices of samples for training, validation, and test
train_ind = 1:95;  %Training
val_ind   = 96:100; %Validation
test_ind  = 101:105; %Test

%Dimension of saved images (which are square)
im_size = 128;

%% No Modifications Past This Point
num_output_images = length(output_idx); %Number of output images

%Throwing error if fig_format doesn't match number of images to load
if make_check_fig
    if (fig_format(1)*fig_format(2)) ~= (num_output_images + 1)
        error('Specified figure display format doesn''t match number images to load.')
    end
end

%Directory names where all data is stored as gneerated from Create_Data.m
tr_v_te_dirs = {sprintf('Training_Data_%d', im_size),...
                sprintf('Validation_Data_%d', im_size),...
                sprintf('Test_Data_%d', im_size)};
            
%Creating figures to visualize that input image and
%corresponding output images match
if make_check_fig
    input_imgs = cell(length(bottle_idx), 1);
    output_imgs = cell(length(bottle_idx), num_output_images);
    full_img_cf = cell(length(bottle_idx), length(im_idx));
    save_dir = 'Data_Check';
    for b_idx = 1:length(bottle_idx)
        %Setting main directory based on index
        if any(bottle_idx(b_idx) == train_ind)
            main_dir = sprintf('Training_Data_%d', im_size);
        elseif any(bottle_idx(b_idx) == val_ind)
            main_dir = sprintf('Validation_Data_%d', im_size);
        else
            main_dir = sprintf('Test_Data_%d', im_size);
        end

        %Current directory
        c_dir = sprintf('Bottle_%d', bottle_idx(b_idx));

        %Loading input images for bottle_idx(ii)
        in_dummy = load(fullfile(pwd, main_dir, c_dir, 'input.mat'));
        input_imgs{b_idx, 1} = in_dummy.input;

        %Loading output images for bottle_idx(ii)
        for out_idx = 1:num_output_images
            out_dummy = load(fullfile(pwd, main_dir, c_dir, sprintf('output_%d.mat', output_idx(out_idx))));
            output_imgs{b_idx, out_idx} = out_dummy.output;
        end

        %Creating figure
        for img = 1:length(im_idx)
            conc_cell = cell(fig_format(1), 1); %Cell to store horizontally concatenated image
            figure(1); clf
            set(gcf, 'color', 'w'); %Sets background of figure white

            %Forming rows by concatenating images according to format size
            for fig_idx1 = 1:fig_format(1)
                if fig_idx1 == 1
                    add_val = 0;
                else
                    add_val = fig_format(2)*(fig_idx1-1);
                end
                for fig_idx2 = 1:fig_format(2)
                    if fig_idx1 == 1 && fig_idx2 == 1
                        conc_cell{fig_idx1} = squeeze(input_imgs{b_idx, 1}(img, :, :));
                    else
                        conc_cell{fig_idx1} = horzcat(conc_cell{fig_idx1}, squeeze(output_imgs{b_idx, add_val + fig_idx2 - 1}(img, :, :)));
                    end
                end
            end

            %Displaying image by concatenation all rows
            full_img = mat2gray(vertcat(conc_cell{:}));
            imshow(full_img, 'InitialMagnification', 3*100) %Show image 3x magnified

            hold on
            [m, n] = size(vertcat(conc_cell{:})); %Size of combined images
            step_x = m/fig_format(1); %Getting period of x for plotting separation lines
            step_y = n/fig_format(2); %Getting period of y for plotting separation lines

            %Plotting white lines to separate images clearly
            for ii = 1:fig_format(1)
                for jj = 1:fig_format(2)
                    plot([(jj-1)*step_x, jj*step_x], ii*[step_y, step_y], 'w-')
                    plot(jj*[step_x, step_x], [(ii-1)*step_y, ii*step_y], 'w-')
                end
            end

            %Saving image is specified to do so
            if save_fig
                [full_img_cf{img}, ~] = frame2im(getframe(gcf)); %Getting figure as image
                if ~exist(fullfile(pwd, save_dir, c_dir), 'dir')
                    mkdir(fullfile(pwd, save_dir, c_dir));
                end

                %Saving image
                imwrite(full_img_cf{img}, fullfile(pwd, save_dir, c_dir, sprintf('data_check_%d.png', img)))          
            end
        end
    end
end

%Throwing error if aug_fig_format doesn't match number of images to load
if make_aug_fig
    if (aug_fig_format(1)*aug_fig_format(2)) ~= length(im_idx_aug)
        error('Specified augmented figure display format doesn''t match number images to load.')
    end
end

%Creating figure to visualize all augmentions
if make_aug_fig
    %Cell to store all images for creating augmented figure
    imgs_aug = cell(length(bottle_idx_aug), length(im_idx_aug));
    
    %Cell to store all created figures showing augmentations
    full_img_cf_aug = cell(length(bottle_idx_aug));
    
    %Directory to save any augmented figures
    save_dir_aug = 'Data_Check_Augmented';
    
    for b_idx = 1:length(bottle_idx_aug)
        %Setting main directory based on index
        if any(bottle_idx_aug(b_idx) == train_ind)
            main_dir = sprintf('Training_Data_%d', im_size);
        elseif any(bottle_idx_aug(b_idx) == val_ind)
            main_dir = sprintf('Validation_Data_%d', im_size);
        else
            main_dir = sprintf('Test_Data_%d', im_size);
        end

        %Current directory
        c_dir = sprintf('Bottle_%d', bottle_idx_aug(b_idx));

        %Loading input images if augmented figure index specifies not to
        %use any output images
        for ii = 1:length(im_idx_aug)
            if ~aug_fig_idx 
                in_dummy = load(fullfile(pwd, main_dir, c_dir, 'input.mat'));
                imgs_aug{b_idx, ii} = squeeze(in_dummy.input(im_idx_aug(ii), :, :));
            else
                out_dummy = load(fullfile(pwd, main_dir, c_dir, sprintf('output_%d.mat', aug_fig_idx)));
                imgs_aug{b_idx, ii} = squeeze(out_dummy.output(im_idx_aug(ii), :, :));
            end
        end
        
        conc_cell = cell(aug_fig_format(1), 1); %Cell to store horizontally concatenated image
        figure(1); clf
        set(gcf, 'color', 'w'); %Sets background of figure white
        
        %Forming rows by concatenating images according to format size
        for ii = 1:aug_fig_format(1)
            if aug_fig_idx == 0
                conc_cell{ii} = horzcat(imgs_aug{b_idx, ((ii-1)*aug_fig_format(2) + 1):(ii*aug_fig_format(2))});
            end
        end
        
        %Displaying image by concatenation all rows
        full_img = mat2gray(vertcat(conc_cell{:}));
        imshow(full_img, 'InitialMagnification', 2*100)

        hold on
        [m, n] = size(vertcat(conc_cell{:})); %Size of combined images
        step_x = m/aug_fig_format(1); %Getting period of x for plotting separation lines
        step_y = n/aug_fig_format(2); %Getting period of y for plotting separation lines
        
        %Plotting white lines to separate images clearly
        for ii = 1:aug_fig_format(1)
            for jj = 1:aug_fig_format(2)
                plot([(jj-1)*step_x, jj*step_x], ii*[step_y, step_y], 'w-')
                plot(jj*[step_x, step_x], [(ii-1)*step_y, ii*step_y], 'w-')
            end
        end
        
        %Saving image is specified to do so
        if save_aug_fig
            [full_img_cf_aug{b_idx}, ~] = frame2im(getframe(gcf)); %Getting figure as image
            if ~exist(fullfile(pwd, save_dir_aug, c_dir), 'dir')
                mkdir(fullfile(pwd, save_dir_aug, c_dir));
            end

            %Saving image
            if ~aug_fig_idx
                imwrite(full_img_cf_aug{b_idx}, fullfile(pwd, save_dir_aug, c_dir, 'augmented_fig_ref.png'))   
            else
                imwrite(full_img_cf_aug{b_idx}, fullfile(pwd, save_dir_aug, c_dir, sprintf('augmented_fig_%d.png', imaug_fig_idxg)))   
            end
        end
    end
end


            