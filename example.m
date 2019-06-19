clear all
clc    
%%%%%%%%%% build your dataset by loading your images%%%%%%
%%%%%%load training data
pathname        = uigetdir;
allfiles        = dir(fullfile(pathname,'*.jpg'));
xtr=[];% initialize training inputs
gamma=[96 97];% size of each image
for i=1:size(allfiles,1)    
x=imread([pathname '\\' allfiles(i).name]);
x=imresize(x,gamma);
x=rgb2gray(x);
x=double(x);
xtr=[xtr; x];% training set building
end
%%%%%%load testing data
pathname        = uigetdir;
allfiles        = dir(fullfile(pathname,'*.jpg'));
xts=[];% initialize testing inputs
for i=1:size(allfiles,1)    
x=imread([pathname '\\' allfiles(i).name]);
x=imresize(x,gamma);
x=rgb2gray(x);
x=double(x);
xts=[xts; x];% testing set building
end

%%%%%%% algorithm parameters%%%%%
NumberofHiddenNeurons=500;
D_ratio=0.35;%the ratio of denoised features from each input
DB=1;% the power of white gaussian noise in decibels 
ActivationFunction='sig';
frame=55;%
%%%% train and test%%%%
[regenerated,corrupted,TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy] = elm_AE(xtr,xts,NumberofHiddenNeurons,ActivationFunction,D_ratio,DB,frame);
subplot(121)
corrupted=corrupted(:,1:gamma(2)*2);
imshow(corrupted')
title('corrupted images ');
subplot(122)
regenerated=regenerated(:,1:gamma(2)*2);
imshow(regenerated')
title('regenerated images');
