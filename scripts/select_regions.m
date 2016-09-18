%% Select regions of interest from mask for all images using Region Growing Algorithm

function [ restImages, J ] = select_regions( restImages, restFiles, proc_mask, maskFile )
    % Select the regions of interest for mask( responsible for disease )
    fileSize = size(restImages, 2);
    figure, imshow(proc_mask); 
    title('Please select regions of interest by clicking on them');
    [ys,xs] = getpts; y=round(ys(1)); x=round(xs(1));
    J = regiongrowing(proc_mask, x, y, 0.2);

    for coord = 2:size(ys,1)
        y = round(ys(coord)); x = round(xs(coord));
        J = regiongrowing(proc_mask, x, y, 0.2) + J;
    end

    figure, imshow(J);
    title(strcat('Positive region images for ', char(maskFile)));


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
end

