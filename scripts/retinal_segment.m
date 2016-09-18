%% Script that takes a mask and processes that image to segment LFAF Retinal
%% sequential images for one eye of a patient.


%% Read in dataset using UI

filter = '*.jpg';
[maskFile, pathname] = uigetfile(fullfile('', filter), 'Select an Initial Mask'); 
maskFile = strcat(pathname, maskFile); 

original = imread(maskFile);  %('OS_Month1_000.jpg') as an example
[cropped_original, rect] = imcrop(original);

% Import the rest of the images for clustering
[restImages, restFiles] = import_images(rect);

% Apply superpixels on image
[image_super_p] = apply_superpixels(cropped_original);

% Merge k-means and active contour into one function
[ segmented_images, restImages, proc_mask ] = k_means_contour( cropped_original, image_super_p, restImages ); 

% Select regions using region growing algorithm for all images
[restImages, J] = select_regions( restImages, restFiles, proc_mask, maskFile );

% Calculate area of regions for all the images and print them
% Create output dialog
fileSize = size(restImages, 2);
messageDialog = cell(1, fileSize + 1);
messageDialog{1} = strcat(num2str(bwarea(J)), ' - Area for ', char(maskFile));

for fileNum = 1:fileSize
    messageDialog{fileNum + 1} = strcat(num2str(bwarea(restImages{fileNum})), ' - Area for ', char(restFiles(fileNum)));
end

msgbox(messageDialog);
