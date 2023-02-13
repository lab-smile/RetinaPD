%Change Remidio Images into a Square
%Maximillian Diaz
%9/25/2019

%This code removes excessive black margins from around the remidio fundus images

clc; clear;

%Move to Directory with Images needing to be processed
cd Normal

%Loops through all images to process them
images=144

for i=1:images
    
name=[num2str(i) '.jpg'];          %Collects image name and stores it in Img
Image=imread(name);              %Loads image as Image

%For data from UKB
%Crop Image to 1376x1376x3 centered around the Fundus Image
%Crop=Image(81:1456, 330:1705,:);

%For data from UF Clinical PD Images
%Crop Image to 2100x2100x3 centered around the Fundus Image
Crop=Image(508:2607,191:2290,:);


imwrite(Crop,name,'jpeg')       %Save image

if mod(i,5)==0
    disp(i)
end

end

