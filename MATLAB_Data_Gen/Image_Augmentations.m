%% Code to Modify all images for Robust Training Data

clear; format short g;

%% User Inputs
%Indices of reference lighting condition images and face images to consider
light_images = 2:36;
bottle_images = 1:105;

%% No Modifications Past This Point
%Number of reference lighting condition images and face images to consider
num_light_images = length(light_images);
num_bottle_images = length(bottle_images);

%Defining directories where data is stored
main_dir = 'Bottle_Images';

%Loading images to alter
relit_images = cell(num_light_images, num_bottle_images); %Cell to store relit images
for ii = 1:num_light_images    
    for jj = 1:num_bottle_images
        %Current directory
        c_dir = sprintf('Bottle_%d', bottle_images(jj));
        
        %Loading relit images
        file_name = fullfile(pwd, 'Relit_Images');
        relit_images{ii, jj} = imread(fullfile(pwd, main_dir, c_dir, sprintf('%d.png', light_images(ii))));
        
    end
end
%Getting image sizes
[m, n] = size(relit_images{1, 1});

%Cells to store altered images
altered_images_xflip = cell(num_light_images, num_bottle_images);
altered_images_yflip = cell(num_light_images, num_bottle_images);
altered_images_xtrans = cell(num_light_images, num_bottle_images, 3, 4);
altered_images_ytrans = cell(num_light_images, num_bottle_images, 3, 4);
altered_images_rotated90 = cell(num_light_images, num_bottle_images, 3, 3);
altered_images_trans = cell(num_light_images, num_bottle_images, 3, 4);

%Altering images by getting rotations of original images and x and y
%flipped images
relit_images_xy = 1;
for ii = 1:num_light_images    
    for jj = 1:num_bottle_images
        %Current directory
        c_dir = sprintf('Bottle_%d', bottle_images(jj));
        
        %Getting current image
        c_im_orig = relit_images{ii, jj};
        
        %Flipping x and y dimensions separately
        altered_images_xflip{ii, jj} = flip(c_im_orig, 2); %Flipping horizontally
        altered_images_yflip{ii, jj} = flip(c_im_orig, 1); %Flipping vertically
        
        c_im_xflip = altered_images_xflip{ii, jj};
        c_im_yflip = altered_images_yflip{ii, jj};
        
        %Saving original image and flipped images
        imwrite(relit_images{ii, jj}, fullfile(pwd, main_dir, c_dir, sprintf('%d_orig_rot0_trans0.png',...
                                     light_images(ii))));
        imwrite(altered_images_xflip{ii, jj}, fullfile(pwd, main_dir, c_dir, sprintf('%d_xflip_rot0_trans0.png',...
                                     light_images(ii))));
        imwrite(altered_images_yflip{ii, jj}, fullfile(pwd, main_dir, c_dir, sprintf('%d_yflip_rot0_trans0.png',...
                                     light_images(ii))));
        
        %Rotating 90 degrees each
        for rot_idx = 1:3
            altered_images_rotated90{ii, jj, 1, rot_idx} = imrotate(c_im_orig, rot_idx*90); %Rotating original image
            altered_images_rotated90{ii, jj, 2, rot_idx} = imrotate(c_im_xflip, rot_idx*90); %Rotating x-flipped image
            altered_images_rotated90{ii, jj, 3, rot_idx} = imrotate(c_im_yflip, rot_idx*90); %Rotating y-flipped image
        
            %Saving rotated original and flipped images
            imwrite(altered_images_rotated90{ii, jj, 1, rot_idx}, fullfile(pwd, main_dir, c_dir, sprintf('%d_orig_rot%d_trans0.png',...
                                         light_images(ii), rot_idx*90)));
            imwrite(altered_images_rotated90{ii, jj, 2, rot_idx}, fullfile(pwd, main_dir, c_dir, sprintf('%d_xflip_rot%d_trans0.png',...
                                         light_images(ii), rot_idx*90)));
            imwrite(altered_images_rotated90{ii, jj, 3, rot_idx}, fullfile(pwd, main_dir, c_dir, sprintf('%d_yflip_rot%d_trans0.png',...
                                         light_images(ii), rot_idx*90)));

        end
        
        %Rotating arbitrary amount
        for trans_idx = 1:2
            %Getting random x and y translations
            if trans_idx == 1
                x_trans = 20; 
                y_trans = 5;
            else
                x_trans = 40; 
                y_trans = 10;
            end
            
            c_im_orig_y_trans   = [imcrop(c_im_orig, [1 y_trans+1 m n]); imcrop(c_im_orig, [1 1 m y_trans-1])];
            c_im_orig_y_trans_n = [imcrop(c_im_orig, [1 n-(y_trans-1) m n]); imcrop(c_im_orig, [1 1 m n-(y_trans+1)])];
            c_im_orig_x_trans   = [imcrop(c_im_orig, [m-(x_trans-1), 1, m n]), imcrop(c_im_orig, [1 1 m-(x_trans+1) n])];
            c_im_orig_x_trans_n = [imcrop(c_im_orig, [x_trans+1 1 m n]), imcrop(c_im_orig, [1 1 x_trans-1 n])];

            altered_images_xtrans{ii, jj, 1, 1+((trans_idx-1)*2)} = c_im_orig_x_trans;
            altered_images_xtrans{ii, jj, 1, 2+((trans_idx-1)*2)} = c_im_orig_x_trans_n;
            altered_images_ytrans{ii, jj, 1, 1+((trans_idx-1)*2)} = c_im_orig_y_trans;
            altered_images_ytrans{ii, jj, 1, 2+((trans_idx-1)*2)} = c_im_orig_y_trans_n;
            
            c_im_xflip_y_trans   = [imcrop(c_im_xflip, [1 y_trans+1 m n]); imcrop(c_im_xflip, [1 1 m y_trans-1])];
            c_im_xflip_y_trans_n = [imcrop(c_im_xflip, [1 n-(y_trans-1) m n]); imcrop(c_im_xflip, [1 1 m n-(y_trans+1)])];
            c_im_xflip_x_trans   = [imcrop(c_im_xflip, [m-(x_trans-1), 1, m n]), imcrop(c_im_xflip, [1 1 m-(x_trans+1) n])];
            c_im_xflip_x_trans_n = [imcrop(c_im_xflip, [x_trans+1 1 m n]), imcrop(c_im_xflip, [1 1 x_trans-1 n])];
            
            altered_images_xtrans{ii, jj, 2, 1+((trans_idx-1)*2)} = c_im_xflip_x_trans;
            altered_images_xtrans{ii, jj, 2, 2+((trans_idx-1)*2)} = c_im_xflip_x_trans_n;
            altered_images_ytrans{ii, jj, 2, 1+((trans_idx-1)*2)} = c_im_xflip_y_trans;
            altered_images_ytrans{ii, jj, 2, 2+((trans_idx-1)*2)} = c_im_xflip_y_trans_n;
            
            c_im_yflip_y_trans   = [imcrop(c_im_yflip, [1 y_trans+1 m n]); imcrop(c_im_yflip, [1 1 m y_trans-1])];
            c_im_yflip_y_trans_n = [imcrop(c_im_yflip, [1 n-(y_trans-1) m n]); imcrop(c_im_yflip, [1 1 m n-(y_trans+1)])];
            c_im_yflip_x_trans   = [imcrop(c_im_yflip, [m-(x_trans-1), 1, m n]), imcrop(c_im_yflip, [1 1 m-(x_trans+1) n])];
            c_im_yflip_x_trans_n = [imcrop(c_im_yflip, [x_trans+1 1 m n]), imcrop(c_im_yflip, [1 1 x_trans-1 n])];
            
            altered_images_xtrans{ii, jj, 3, 1+((trans_idx-1)*2)} = c_im_yflip_x_trans;
            altered_images_xtrans{ii, jj, 3, 2+((trans_idx-1)*2)} = c_im_yflip_x_trans_n;
            altered_images_ytrans{ii, jj, 3, 1+((trans_idx-1)*2)} = c_im_yflip_y_trans;
            altered_images_ytrans{ii, jj, 3, 2+((trans_idx-1)*2)} = c_im_yflip_y_trans_n;
        end
        
        for idx = 1:3
            if idx == 1
                im_string = 'orig';
            elseif idx == 2
                im_string = 'xflip';
            else
                im_string = 'yflip';
            end
            for trans_idx = 1:2
                %Saving rotated original and flipped images with translations
                imwrite(altered_images_xtrans{ii, jj, idx, 1+((trans_idx-1)*2)}, fullfile(pwd, main_dir, c_dir, sprintf('%d_%s_rot0_trans_x%d_p.png',...
                                             light_images(ii), im_string, trans_idx)));
                imwrite(altered_images_xtrans{ii, jj, idx, 2+((trans_idx-1)*2)}, fullfile(pwd, main_dir, c_dir, sprintf('%d_%s_rot0_trans_x%d_n.png',...
                                             light_images(ii), im_string, trans_idx)));
                imwrite(altered_images_ytrans{ii, jj, idx, 1+((trans_idx-1)*2)}, fullfile(pwd, main_dir, c_dir, sprintf('%d_%s_rot0_trans_y%d_p.png',...
                                             light_images(ii), im_string, trans_idx)));
                imwrite(altered_images_ytrans{ii, jj, idx, 2+((trans_idx-1)*2)}, fullfile(pwd, main_dir, c_dir, sprintf('%d_%s_rot0_trans_y%d_n.png',...
                                             light_images(ii), im_string, trans_idx)));
            end
        end        
    end
    fprintf('%d/%d lighting images completed!\n', ii, num_light_images)
end