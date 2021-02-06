clc; clear;

cd Balanced
cd Deidentified

list=dir;
list=(struct2cell(list));
list=list(1,3:end);
number=length(list);

labels=zeros(147,1);
%labels(1)=9;
for i=1:1:number
    
Img=cell2mat(list(i));          %Collects image name and stores it in Img
Image=imread(Img);              %Loads image as Image


name=[num2str(i) '.jpg'];

if Img(end-4)=='X'
    labels(i)=1;
else
    truth=str2num(Img(end-4));
    labels(i)=truth;
end 

imwrite(Image,name,'jpeg')       %Save image

if mod(i,5)==0
    disp(i)
end

end
