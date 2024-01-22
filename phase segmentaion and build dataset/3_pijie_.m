clc
clear
a2 = struct2cell(load('test_1.mat')); 
a3 = struct2cell(load('test_2.mat'));
a4 = struct2cell(load('test_3.mat'));
a5 = struct2cell(load('test_4.mat'));
a6 = struct2cell(load('test_5.mat'));
a7 = struct2cell(load('test_6.mat'));
% a8 = struct2cell(load('test_7.mat'));

all = [a2 a3 a4 a5 a6  a7];          
test = [all{1,1} all{1,2} all{1,3} all{1,4} all{1,5}  all{1,6}]; 
save('test.mat','test');


% clc
% clear
% a2 = struct2cell(load('train_1.mat')); 
% a3 = struct2cell(load('train_2.mat'));
% a4 = struct2cell(load('train_3.mat'));
% a5 = struct2cell(load('train_4.mat'));
% a6 = struct2cell(load('train_5.mat'));
% a7 = struct2cell(load('train_6.mat'));
% a8 = struct2cell(load('train_7.mat'));
% all = [a2 a3 a4 a5 a6 a7 ];         
% train = [all{1,1} all{1,2} all{1,3} all{1,4} all{1,5} all{1,6} ];  
% save('train.mat','train');



