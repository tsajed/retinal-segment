
%% Evaluate Model using JACCARD or DICE scores


%% Read in dataset using UI

eval_coords = import_true_labels();

filter = '*.jpg';
[maskFile, pathname] = uigetfile(fullfile('', filter), 'Select an Initial Mask'); % Get mask for superpixalation
maskFile = strcat(pathname, maskFile); 

im = imread(maskFile);  %('OS_Month1_000.jpg') as an example
[I2, rect] = imcrop(im);

[restImages, restFiles] = import_images(rect);
fileSize = size(restImages, 2);

[labels,N] = superpixels(I2,300);  % Find superpixels
outputImage = zeros(size(I2),'like',I2);
idx2 = label2idx(labels);

% Average out the pixels from original image that belongs to same cluster superpixel 
for labelVal = 1:N
    greyIdx = idx2{labelVal};
    outputImage(greyIdx) = mean(I2(greyIdx));
end

m = rgb2gray(outputImage);
cropped_super = double( m(:,:,1) );

figure(2);
imshow(m, []);
m = double( m(:,:,1) );

% Merge k-means and active contour into one function
[ segmented_images, restImages, proc_mask ] = k_means_contour( I2, m, restImages); 

% Select regions using region growing algorithm for all images
[restImages, J] = select_regions( restImages, restFiles, proc_mask, maskFile );

% Calculate DICE scores for all the images and mask
dice_scores = calc_dice_scores( J, rect, eval_coords, restImages, im);

% Print the dice score values and area of region
bwarea(J)
dice_scores{1}

for fileNum = 1:fileSize
    bwarea(restImages{fileNum})
    dice_scores{fileNum + 1}
end
