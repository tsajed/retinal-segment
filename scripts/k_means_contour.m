%% Apply k-means algorithm on mask with k = 2 using both superpixelated and original mask
   % Then apply active contours on the mask and rest of images

function [ segmented_images, restImages, proc_mask ] = k_means_contour( cropped_original, m , restImages)
    % Calculate features for k-means algorithm. 
    % Feature 1 - Intensity of superpixels for x,y
    % Feature 2 - Sum of Regional intensity for (x,y)
    
    f = double( cropped_original(:,:,1) );
    nrows = size(f, 1);
    ncols = size(f, 2);
    K = 2;
    features = zeros(nrows*ncols, 2);
    fileSize = size(restImages, 2);

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

    [idx, C] = kmeans(norm_features, K, 'MaxIter', 300); % K-means, k = 2
    idx = reshape(idx, nrows, ncols);
    mask = idx;
    imshow(idx, []);

    segmented_images = cell(1,2);
    rgb_label = repmat(idx,[1 1 3]);
    for k = 1:K
        segment = cropped_original;
        segment(rgb_label ~= k) = 0;
        segmented_images{k} = segment;
        if k == 2
            mask(idx == k) = 0;
        end
    end
    
    % ActiveContour using mask from output of k-means cluster, smoothed to
    % retain regions
    maxIterations = 200; 
    proc_mask = activecontour(f, mask, maxIterations, 'Chan-Vese', 'SmoothFactor', 5.0);

    % Do it on all the other images of eye
    for fileNum = 1:fileSize
        restImages{fileNum} = activecontour(restImages{fileNum}, mask, maxIterations, 'Chan-Vese', 'SmoothFactor',5.0);
    end
end

