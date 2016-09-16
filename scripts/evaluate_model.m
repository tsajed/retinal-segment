
%% Evaluate Model using JACCARD or DICE scores

%% Set up some parameters

K = 2; % number of classes try 4, 5 or 6

%% Read in dataset using UI
filter = '*.txt';

% Get all labelled coordinates from UI
[evalFiles, pathname] = uigetfile(fullfile('', filter), 'Select labelled maps for evaluation', 'MultiSelect', 'on');
evalfileSize = size(evalFiles, 2);
eval_coords = cell(1,5);
binary_true_labels = cell(1,5);
dice_scores = cell(1,5);

for fileNum = 1:evalfileSize
    evalFiles(fileNum) = strcat(pathname, evalFiles(fileNum));
    eval_coords{fileNum} = importdata(char(evalFiles(fileNum)));
end

filter = '*.jpg';
[maskFile, pathname] = uigetfile(fullfile('', filter), 'Select an Initial Mask'); % Get mask for superpixalation
maskFile = strcat(pathname, maskFile); 
[restFiles, pathname] = uigetfile(fullfile('', filter), 'Select sequential images for the same eye', 'MultiSelect', 'on'); % Get all other images for the eye

fileSize = size(restFiles, 2);
for fileNum = 1:fileSize
    restFiles(fileNum) = strcat(pathname, restFiles(fileNum));
end

im = imread(maskFile);  %('OS_Month1_000.jpg') as an example
[I2, rect] = imcrop(im);

% Create true label from polygon mask
for i = 1:evalfileSize
    binary_true_temp = poly2mask(eval_coords{i}(:,1), eval_coords{i}(:,2), size(im, 1), size(im, 2));
    binary_true_labels{i} = imcrop(binary_true_temp, rect);
end


% Crop the rest of the images same dimension as the mask
restImages = cell(1,5);
for fileNum = 1:fileSize
    restImages{fileNum} = imread(char(restFiles(fileNum)));
    restImages{fileNum} = imcrop(restImages{fileNum}, rect);
    tempImg = restImages{fileNum};
    restImages{fileNum} = double( tempImg(:,:,1) );
end

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
bw = activecontour(f, mask, maxIterations, 'Chan-Vese', 'SmoothFactor',5.0);

% Do it on all the other images of eye
for fileNum = 1:fileSize
    restImages{fileNum} = activecontour(restImages{fileNum}, mask, maxIterations, 'Chan-Vese', 'SmoothFactor',5.0);
end


% Select the regions of interest ( responsible for disease )
figure, imshow(bw); 
title('Please select regions of interest by clicking on them');
[ys,xs] = getpts; y=round(ys(1)); x=round(xs(1));
J = regiongrowing(bw, x, y, 0.2);

for coord = 2:size(ys,1)
    y = round(ys(coord)); x = round(xs(coord));
    J = regiongrowing(bw, x, y, 0.2) + J;
end

figure, imshow(J);
title(strcat('Positive region images for ', char(maskFile)));

% Find DICE Scores for mask
dice_scores{1} = 2*nnz(J&binary_true_labels{1})/(nnz(J) + nnz(binary_true_labels{1}));

% Using the same points, select regions for the rest of the images
for fileNum = 1:fileSize
    tempImage = regiongrowing(restImages{fileNum}, round(xs(1)), round(ys(1)), 0.2);
    for coord = 2:size(ys,1)
        tempImage = regiongrowing(restImages{fileNum}, round(xs(coord)), round(ys(coord)), 0.2) + tempImage;
    end
    figure, imshow(tempImage);
    title(strcat('Positive region images for ', char(restFiles(fileNum))));
    restImages{fileNum} = tempImage;
end

% Find DICE scores for rest of images
for i = 1:fileSize
    dice_scores{i+1} = 2*nnz(restImages{i}&binary_true_labels{i+1})/(nnz(restImages{i}) + nnz(binary_true_labels{i+1}));
end

% Calculate area of regions for all the images and print them
bwarea(J)
dice_scores{1}

for fileNum = 1:fileSize
    bwarea(restImages{fileNum})
    dice_scores{fileNum + 1}
end
