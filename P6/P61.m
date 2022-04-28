
%% Lab 6.1: PCA 

clear all
close all

% load images 
% images size is 20x20. 

load('MNISTdata2.mat'); 

nrows=20;
ncols=20;

nimages = size(X,1);

%Show the images
for I=1:40:nimages, 
    imshow(reshape(X(I,:),nrows,ncols))
    pause(0.1)
end


%% Perform PCA following the instructions of the lab


%% Use the classifier from previous labs on the projected space







