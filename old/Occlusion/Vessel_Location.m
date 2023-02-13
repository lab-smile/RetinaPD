%cd 'Green_Bin'

%Vessel Map to Binary Code
%SMiLE Lab, 10/24/2019
%Maximillian Diaz
clc; clear;

cd Green

name=[num2str(391) 'small.png'];           %Load Image
Img=imread(name); 

imshow(Img)

A=Img;              %Create Binary Using predetermined threshold
A=A(:,:,1);
A(A<60)=0;
A(A>=60)= 1;
A=logical(A);

figure(2)
imshow(A)

[row, col]=find(A);

row=row-1;
col=col-1;

Q=[row,col];
xlswrite('391index.xlsx', Q)
