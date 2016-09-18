
%% Evaluate Model using JACCARD or DICE scores


%% Read in dataset using UI

filter = '*.jpg';
[maskFile, pathname] = uigetfile(fullfile('', filter), 'Select an Initial Mask'); 
maskFile = strcat(pathname, maskFile); 

original = imread(maskFile);  %('OS_Month1_000.jpg') as an example
[cropped_original, rect] = imcrop(original);

% Import the rest of the images for clustering
[restImages, restFiles] = import_images(rect);
eval_coords = import_true_labels();

% Apply superpixels on image
[image_super_p] = apply_superpixels(cropped_original);

% Merge k-means and active contour into one function
[ segmented_images, restImages, proc_mask ] = k_means_contour( cropped_original, image_super_p, restImages ); 

% Select regions using region growing algorithm for all images
[restImages, J] = select_regions( restImages, restFiles, proc_mask, maskFile );

% Calculate DICE scores for all the images and mask
dice_scores = calc_dice_scores( J, rect, eval_coords, restImages, original);

% Print the dice score values and area of regions for all images
bwarea(J)
dice_scores{1}

for fileNum = 1:size(restImages, 2)
    bwarea(restImages{fileNum})
    dice_scores{fileNum + 1}
end
