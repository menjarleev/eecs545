%% Code to view and save montage of bottle images for easy viewing of all lighting conditions
clear; format short g

%% User Inputs
%Flag to store generated images
save_data = 1;

%Indices of lighting angles to load
light_idx = 2:36;

%Formats to display all images (array size)
fig_format = [5, 7];

%Additionally suffix for loading a specific set of augmented images
aug_img_str = '';

%% No Modifications Past This Point
%Throwing error if fig_format doesn't match number of images to load
if (fig_format(1)*fig_format(2)) ~= length(light_idx)
    error('Specified figure display format doesn''t match number images to load.')
end

%Initializing cell store images
all_imgs = cell(length(light_idx), 1); %Stores all individual images

%Loading all bottle images
main_dir = 'VolCNN';
main_dir_2 = 'VolCNN_Comb';

%Loading all lighting condition images
for l_loop = 1:length(light_idx)
    all_imgs{l_loop} = imrotate(imread(fullfile(pwd, main_dir, sprintf('%d%s.png', light_idx(l_loop), aug_img_str))), 90);
end

%Creating figure
conc_cell = cell(fig_format(1)); %Cell to store horizontally concatenated image
figure(1); clf
set(gcf, 'color', 'w'); %Sets background of figure white

%Forming rows by concatenating images according to format size
for ii = 1:fig_format(1)
    conc_cell{ii} = horzcat(all_imgs{((ii-1)*fig_format(2) + 1):(ii*fig_format(2))});
end

%Displaying image by concatenation all rows
full_img = vertcat(conc_cell{:});
imshow(full_img)
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
if save_data
    [full_img_cf, ~] = frame2im(getframe(gcf)); %Getting figure as image
    if strcmp(aug_img_str, '')
        imwrite(full_img_cf, fullfile(pwd, main_dir, 'comb_def.png'))          
    else
        imwrite(full_img_cf, fullfile(pwd, main_dir, sprintf('Comb%s.png', aug_img_str)))
    end

    if ~exist(fullfile(pwd, main_dir_2), 'dir')
        mkdir(fullfile(pwd, main_dir_2))
    end
    if strcmp(aug_img_str, '')
        imwrite(full_img_cf, fullfile(pwd, main_dir_2, 'VolCNNBottle_Comb_def.png'))          
    else
        imwrite(full_img_c, fullfile(pwd, main_dir_2, sprintf('VolCNNBottle_Comb%s.png', aug_img_str)))
    end

end

%Closing figure
close all