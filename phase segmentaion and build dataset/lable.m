clc
clear
XSENS_path = 'C:\Users\Administrator\Desktop\cx\相位划分\5.29\trainlable - 副本 (2)\';
XSENS_name = '*.xlsx';
XSENSPath = dir([XSENS_path, XSENS_name]);
[trainlabels]=xlsread(strcat(XSENS_path ,XSENSPath (1).name));
  save('trainlabels.mat','trainlabels');

% clc
% clear
% XSENS_path = 'C:\Users\Administrator\Desktop\cx\相位划分\5.29\testlable - 副本 (2)\';
% XSENS_name = '*.xlsx';
% XSENSPath = dir([XSENS_path, XSENS_name]);
% [testlabels]=xlsread(strcat(XSENS_path ,XSENSPath (1).name));
% save('testlabels.mat','testlabels');