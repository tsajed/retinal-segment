
%% Create polygon from coordinates and calculate DICE scores 

function dice_scores = calc_dice_scores( mask, cropped_rect, eval_coords, rest_images, original_image )
    % Create true label from polygon mask
    eval_size = size(eval_coords, 2);
    binary_true_labels = cell(1, eval_size);
    dice_scores = cell(1,eval_size);
    
    for i = 1:eval_size
        binary_true_temp = poly2mask(eval_coords{i}(:,1), eval_coords{i}(:,2), size(original_image, 1), size(original_image, 2));
        binary_true_labels{i} = imcrop(binary_true_temp, cropped_rect);
    end
    
    % Find DICE Scores for mask
    dice_scores{1} = 2*nnz(mask&binary_true_labels{1})/(nnz(mask) + nnz(binary_true_labels{1}));
    
    % Find DICE scores for rest of images
    for i = 1:(eval_size - 1)
        dice_scores{i+1} = 2*nnz(rest_images{i}&binary_true_labels{i+1})/(nnz(rest_images{i}) + nnz(binary_true_labels{i+1}));
    end
end

