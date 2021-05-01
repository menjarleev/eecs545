clear; format short g

%% User Inputs
%Flag to store generated images
save_data = 1;

%Bottle index to rotate by 90 degrees and regenerate images
bottle_idx = 18;

%Lighting directions
L = [-1, 0,    0;
     1,  0,    0;
     1,  0.8,  0;
     1, -0.8,  0;
     1,  0.8,  0.5;
     1, -0.8,  0.5;
     1,  0,   -0.5;
     1,  0.8, -0.5;
     1, -0.8, -0.5;
     1,  0.8,  0.8;
     1, -0.8,  0.8;
     1,  0,   -0.8;
     1,  0.8, -0.8;
     1, -0.8, -0.8;
     1,  0.5,  0;
     1, -0.5,  0;
     1,  0.5,  0.5;
     1, -0.5,  0.5;
     1,  0.5, -0.5;
     1, -0.5, -0.5;
     1,  0.5,  0.8;
     1, -0.5,  0.8;
     1,  0.5, -0.8;
     1, -0.5, -0.8;
     1,  0,    0.5;
     1,  0,    0.8;
     1, -1,   -0.8;
     1, -1,   -0.5;
     1, -1,    0;
     1, -1,    0.5;
     1, -1,    0.8;
     1,  1,   -0.8;
     1,  1,   -0.5;
     1,  1,    0;
     1,  1,    0.5;
     1,  1,    0.8];
 
L = [L(:, 2), L(:, 1), L(:, 3)];

%% No Modifications Past This Point
%Directory to store images
main_dir = 'Bottle_Images';
if ~exist(main_dir, 'dir')
    mkdir(main_dir)
end
 
for img = bottle_idx
    %Defining directory to store images for specific bottle
    c_img_dir = sprintf('Bottle_%d', img);
    if ~exist(fullfile(pwd, main_dir, c_img_dir), 'dir')
        mkdir(fullfile(pwd, main_dir, c_img_dir))
    end
    
    %Loading stl file
    fv = stlread(fullfile(pwd, 'STL_Files', sprintf('bottle_%d.stl', img)));

    %Initializing matrix to store images
    I = zeros(256, 256, size(L, 1));

    for ii = 1:size(L, 1)
        figure('Position', [400 400 246/2 246/2]); clf
        patch(fv,'FaceColor', [1, 1, 1], ...
                 'EdgeColor', 'none', ...
                 'FaceLighting', 'flat', ...
                 'AmbientStrength', 0, ...
                 'DiffuseStrength', 0, ...
                 'SpecularStrength', 0);

        %Add single point light source, and tone down the specular highlighting
        material('dull');
        light('Position', L(ii, :), 'Style', 'infinite');

        %Adjusting axis and background
        axis('image'); %Setting to make figure show object correctly
        set(gca, 'visible', 'off') %Making axis invisible
        set(gcf, 'InvertHardCopy', 'off'); %Turning axis background off
        set(gcf, 'Color',[0 0 0]); %Set background to [0 0 0] to make black

        view([180 0]); %Setting appropriate viewing angle
        saveas(gcf, fullfile(pwd, main_dir, c_img_dir, sprintf('%d.png', ii))); % save as .png file 
        close all

        %Loading saved image and ensuring it's 256x256, and if not, resizing it
        %and saving it over the current image
        c_img = imread(fullfile(pwd, main_dir, c_img_dir, sprintf('%d.png', ii)));
        [m, n, k] = size(c_img);
        if ii == 1
            sub_color = [max(max(c_img(:, :, 1))),...
                         max(max(c_img(:, :, 2))),...
                         max(max(c_img(:, :, 3)))];
        end
        %Resizing if needed and subtracting ambient color
        for kk = 1:3
            c_img(:, :, kk) = imresize(c_img(:, :, kk), [256, 256]) - sub_color(kk);
        end
        imwrite(imadjust(rgb2gray(c_img)), fullfile(pwd, main_dir, c_img_dir, sprintf('%d.png', ii))); % save as .png file

        %Converting to grayscale and then double
        c_img_d = im2double(rgb2gray(c_img));

        %Storing in image matrix
        I(:, :, ii) = c_img_d;
    end

    %Determining mask 
    M = imfill(imbinarize(sum(I(:, :, 2:end), 3)), 'holes');

    %Saving in format used for photometric stereo
    if save_data == 1
        %Creating struct
        bottle_xy.I = I(:, :, 2:end);
        bottle_xy.L = L(2:end, :)';
        bottle_xy.M = M;

        save(sprintf('bottle_xy_%d', img), '-struct', 'bottle_xy');
    end
end