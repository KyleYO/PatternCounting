img=imread('C:\Users\������\Desktop\�פ�i��\������\�Ϯw\re_1.jpg');
imshow(img);  
[m n]=size(img);  
img=double(img);  
%%canny????���e?�B��?���`?�A�ҥH�ڴN����?�Ψt?��?�F  
%%����?�i  
w=fspecial('gaussian',[5 5]);  
img=imfilter(img,w,'replicate');  
figure;  
imshow(uint8(img))  
%%sobel????  
w=fspecial('sobel');  
img_w=imfilter(img,w,'replicate');      %�D???  
w=w';  
img_h=imfilter(img,w,'replicate');      %�D???  
img=sqrt(img_w.^2+img_h.^2);        %�`�N?�����O??���D�����A�ӬO����M�b?��C�ڴ�?�n?�@�q??���d?�F  
figure;  
imshow(uint8(img))  
% %%�U���O�D��j���  
% new_edge=zeros(m,n);  
% for i=2:m-1  
%     for j=2:n-1  
%         Mx=img_w(i,j);  
%         My=img_h(i,j);  
%           
%         if My~=0  
%             o=atan(Mx/My);      %??���k?����  
%         elseif My==0 && Mx>0  
%             o=pi/2;  
%         else  
%             o=-pi/2;              
%         end  
%           
%         %Mx?��My�Mimg?�洡��  
%         adds=pioter_coords(o);%??�����k?�@?�D�o��??��?�A���Ȼݭn         
%         M1=My*img(i+adds(2),j+adds(1))+(Mx-My)*img(i+adds(4),j+adds(3));   %���ȦZ�o�쪺�����A�Φ������M?�e������?   
%         adds=pioter_coords(o+pi);%??�k?�t�@?�D�o��??��?�A���Ȼݭn  
%         M2=My*img(i+adds(2),j+adds(1))+(Mx-My)*img(i+adds(4),j+adds(3));%�t�@?���ȱo�쪺�����A�P?�M?�e������?  
%           
%         isbigger=(Mx*img(i,j)>M1)*(Mx*img(i,j)>=M2)+(Mx*img(i,j)<M1)*(Mx*img(i,j)<=M2); %�p�G?�e?��???���j�m1  
%           
%         if isbigger  
%            new_edge(i,j)=img(i,j);   
%         end          
%     end  
% end  
% figure;  
% imshow(uint8(new_edge))  
% %%�U���O?�Z?��?�z  
% up=120;     %�W?��  
% low=100;    %�U?��  
% set(0,'RecursionLimit',10000);  %?�m�̤j??�`��  
% for i=1:m  
%     for j=1:n  
%       if new_edge(i,j)>up &&new_edge(i,j)~=255  %�P?�W?��  
%             new_edge(i,j)=255;  
%             new_edge=pioter_connect(new_edge,i,j,low);  
%       end  
%     end  
% end  
% figure;  
% imshow(new_edge==255)  

