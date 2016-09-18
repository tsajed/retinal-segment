
%% Import true labels from manually labelled images

function eval_coords = import_true_labels()
    filter = '*.txt';
    
    % Get all labelled coordinates from UI
    [evalFiles, pathname] = uigetfile(fullfile('', filter), 'Select labelled maps for evaluation', 'MultiSelect', 'on');
    evalfileSize = size(evalFiles, 2);
    eval_coords = cell(1,evalfileSize);
    
    % Save coordinates in eval_coords
    for fileNum = 1:evalfileSize
    evalFiles(fileNum) = strcat(pathname, evalFiles(fileNum));
    eval_coords{fileNum} = importdata(char(evalFiles(fileNum)));
    end
end