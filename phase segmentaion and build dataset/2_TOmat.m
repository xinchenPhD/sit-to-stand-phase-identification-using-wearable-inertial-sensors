clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\5.29\test_7\';   

XSENS_name = '*.xlsx';
XSENSPath = dir([XSENS_path, XSENS_name]);
Length = length(XSENSPath );    
for i = 1:Length
    [data]=xlsread(strcat(XSENS_path ,XSENSPath (i).name))
    c = size(data,1)
    a = mat2cell(data',size(data,2),size(data,1))
    d(1,i) = a
    x = strsplit(XSENSPath (i).name,'.')
    x1 =cell2mat(x(1,1))
    x2 = strcat(XSENS_path,x1,'.mat')
    save(x2,'a');
    if  i == Length
         save('test_7.mat','d');     
    end  
end
