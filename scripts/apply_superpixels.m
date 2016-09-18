%% Superpixal on mask, and show that image, return that superpixelated image in double

function [ image_super_p ] = apply_superpixels( cropped_original )

    [labels,N] = superpixels(cropped_original, 300);  % Find superpixels
    outputImage = zeros(size(cropped_original),'like',cropped_original);
    idx2 = label2idx(labels);

    % Average out the pixels from original image that belongs to same cluster superpixel 
    for labelVal = 1:N
        greyIdx = idx2{labelVal};
        outputImage(greyIdx) = mean(cropped_original(greyIdx));
    end

    image_super_p = rgb2gray(outputImage);

    figure(2);
    imshow(image_super_p, []);
    image_super_p = double( image_super_p(:,:,1) );
end

