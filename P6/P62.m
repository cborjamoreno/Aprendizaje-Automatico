%% Lab 6.2: SVD 

clear all
close all

% use YOUR image!
I = imread('cesar.jpg');

% Convert to B&W
BW = rgb2gray(I);

% Convert data to double
X=im2double(BW);

% show image
figure(1);
colormap(gray);
imshow(X);
axis off;
pause

% Apply SVD

% Plot first 5 components
for k = 1:5
    figure(2);
    %imshow(Xhat);
    colormap(gray);
    axis off;
    pause
end

% Plot the image reconstructed with 1, 2, 5, 10, 20, and the total number
% of components
for k = [1 2 5 10 20 rank(X)],
    figure(3);
    %imshow(Xhat);
    colormap(gray);
    axis off;
    pause
end

% Find the value of k that maintains 90% of variability

% Plot the image reconstructed with the first  k components

% Compute and show savings in space
