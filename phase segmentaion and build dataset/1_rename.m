clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\TEMP\';
XSENS_name = '*.xlsx';
XSENSPath = dir([XSENS_path, XSENS_name]);
Length = length(XSENSPath );    
for i = 1:Length
    [data]=xlsread(strcat(XSENS_path ,XSENSPath (i).name));
    x = num2str(i);
    x2 = strcat(XSENS_path,x ,'_1' ,'.xlsx')
    xlswrite(x2 , data);
end
