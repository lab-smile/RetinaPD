%Grey
%SMiLE Lab, 10/24/2019
%Maximillian Diaz

%This code creates a grey scale iamge of the original fundus images


clc; clear


images=144

cd Green

for i=1:images
    
name=[num2str(i) '.jpg'];          %Collects image name and stores it in Img
Img=imread(name);              %Loads image as Image


G = Img(:,:,2);                %Used to isolate green channel


G=cat(3,G,G,G);
imwrite(G,name,'jpg')       %Save image

if mod(i,5)==0
    disp(i)
end    


end


