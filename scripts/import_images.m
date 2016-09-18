%% Function that imports images to be segmented in sequence for an eye

function [ restImages, restFiles ] = import_images(cropped_rect)
    filter = '*.jpg';
    [restFiles, pathname] = uigetfile(fullfile('', filter), 'Select sequential images for the same eye', 'MultiSelect', 'on'); % Get all other images for the eye

    fileSize = size(restFiles, 2);
    restImages = cell(1, fileSize);
    
    for fileNum = 1:fileSize
        restFiles(fileNum) = strcat(pathname, restFiles(fileNum));
    end

    % Crop the rest of the images same dimension as the mask
    for fileNum = 1:fileSize
        restImages{fileNum} = imread(char(restFiles(fileNum)));
        restImages{fileNum} = imcrop(restImages{fileNum}, cropped_rect);
        tempImg = restImages{fileNum};
        restImages{fileNum} = double( tempImg(:,:,1) );
    end

end

