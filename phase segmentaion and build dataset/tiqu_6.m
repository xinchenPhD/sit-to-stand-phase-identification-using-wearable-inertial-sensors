clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\test\test_3\';
save_path = 'C:\Users\Administrator\Desktop\cx\test\test_3_gyr\';
XSENS_name = '*.xlsx';
XSENSPath = dir([XSENS_path, XSENS_name]);
Length = length(XSENSPath );  
for i = 1:Length
    [data]=xlsread(strcat(XSENS_path ,XSENSPath (i).name))
    data_6 = data(:,2); 
    x = strsplit(XSENSPath (i).name,'.')
    x1 =cell2mat(x(1,1))
    x2 = strcat(save_path,x1,'.xlsx')
    xlswrite(x2 , data_6);
end
