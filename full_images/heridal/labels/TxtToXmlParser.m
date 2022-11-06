% fileID = fopen('IMG_3942b.txt','r');
% formatSpec = '%f';
% sizeA = [16 Inf];
% A = fscanf(fileID,formatSpec,sizeA)

Files=dir('C:/Users/zeljko/Dropbox/Deep Learning Approach for SAR/Image Database/DronImages/Medjugorje/Medjugorje Brdo ukazanja 1/New folder/labels/*.txt');
for k=1:length(Files)
    FileNames=Files(k).name
    [pathstr,name,ext] = fileparts(FileNames)
    fileID = fopen(strcat('C:/Users/zeljko/Dropbox/Deep Learning Approach for SAR/Image Database/DronImages/Medjugorje/Medjugorje Brdo ukazanja 1/New folder/labels/',FileNames),'r')
    formatSpec = '%f';
    sizeA = [16 Inf];
    Ab = fscanf(fileID,formatSpec,sizeA);
    Ab
   
    CreateXML(name,'human',Ab);
end

%xmlCreator(name,'human',Ab(:,1));