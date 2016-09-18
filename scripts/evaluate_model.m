
%% Evaluate Model using JACCARD or DICE scores

%% Set up some parameters

K = 2; % number of classes try 4, 5 or 6

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

f = double( I2(:,:,1) );
nrows = size(f, 1);
ncols = size(f, 2);

m = rgb2gray(outputImage);
cropped_super = double( m(:,:,1) );

figure(2);
imshow(m, []);
m = double( m(:,:,1) );

% Calculate features for k-means algorithm. 
% Feature 1 - Intensity of superpixels for x,y
% Feature 2 - Sum of Regional intensity for (x,y)

features = zeros(nrows*ncols, 2);

iterator = 0;
for i = 1:ncols
    for j = 1:nrows
        iterator = iterator + 1;
        features(iterator, 1) = m(j,i);
        sum = 0;
        for a = i-10:i+10 % for less noise and more cluster
            for b = j-10:j+10
                if a <= ncols && b <= nrows && a >= 1 && b >= 1
                    sum = sum + m(b,a);
                end
            end
        end
        features(iterator, 2) = sum;
    end
end

norm_features = normc(features);

[idx, C] = kmeans(norm_features, K, 'MaxIter', 300);
idx = reshape(idx, nrows, ncols);
mask = idx;
imshow(idx, []);


segmented_images = cell(1,2);
rgb_label = repmat(idx,[1 1 3]);
for k = 1:K
    segment = I2;
    segment(rgb_label ~= k) = 0;
    segmented_images{k} = segment;
    if k == 2
        mask(idx == k) = 0;
    end
end

% ActiveContour using mask from output of k-means cluster, smoothed to
% retain regions
maxIterations = 200; 
bw = activecontour(f, mask, maxIterations, 'Chan-Vese', 'SmoothFactor', 5.0);

% Do it on all the other images of eye
for fileNum = 1:fileSize
    restImages{fileNum} = activecontour(restImages{fileNum}, mask, maxIterations, 'Chan-Vese', 'SmoothFactor',5.0);
end

% Select regions using region growing algorithm for all images
[restImages, J] = select_regions( restImages, restFiles, bw, maskFile );

% Calculate DICE scores for all the images and mask
dice_scores = calc_dice_scores( J, rect, eval_coords, restImages, im);

% Print the dice score values and area of region
bwarea(J)
dice_scores{1}

for fileNum = 1:fileSize
    bwarea(restImages{fileNum})
    dice_scores{fileNum + 1}
end
