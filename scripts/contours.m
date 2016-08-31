

im = imread('cropped_OS_M1_2.jpg');

% im = imnoise(im,'gaussian');
[I2, rect] = imcrop(im);  % cropping the image

[labels,N] = superpixels(I2,100); % Find superpixels

outputImage = zeros(size(I2),'like',I2);
idx2 = label2idx(labels);

for labelVal = 1:N
    greyIdx = idx2{labelVal};
    outputImage(greyIdx) = mean(I2(greyIdx));
end

m = outputImage; %rgb2gray(outputImage);
imshow(m, []);

  
str = 'Click to select initial contour location. Double-click to confirm and proceed.';
title(str,'Color','b','FontSize',12);
disp(sprintf('\nNote: Click close to object boundaries for more accurate result.'))

mask = roipoly;
  
figure, imshow(mask)
title('Initial MASK');

maxIterations = 200; 
bw = activecontour(m, mask, maxIterations, 'Chan-Vese');
  
% Display segmented image
figure, imshow(bw)
title('Active Contours for Patient 2');
