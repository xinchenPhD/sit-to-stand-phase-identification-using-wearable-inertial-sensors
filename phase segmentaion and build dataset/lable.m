clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\5.29\';
XSENS_name = '*.xlsx';
XSENSPath = dir([XSENS_path, XSENS_name]);
[trainlabels]=xlsread(strcat(XSENS_path ,XSENSPath (1).name));
  save('trainlabels.mat','trainlabels');

