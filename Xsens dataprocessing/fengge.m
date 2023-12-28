clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\';  
XSENS_name = '*.xlsx';

XSENSPath = dir([XSENS_path, XSENS_name]);
Length = length(XSENSPath );    
[data]=xlsread(strcat(XSENS_path ,XSENSPath (3).name));
c = size(data,1);

for i = 1:100 :c
    segment_filename = sprintf('%d.xlsx', i );
    input100 = data(i:i+99,:);
    xlswrite(segment_filename , input100);
end





