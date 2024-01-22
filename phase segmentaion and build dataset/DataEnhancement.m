clc;clear;
clear all;

data_expand_scale = 10;

ori_data = load('./sts_2-3-4-5-6.mat');
trainData = ori_data.sts.train;
trainDataLabels = ori_data.sts.trainlabels;
testData = ori_data.sts.test;
testDataLabels = ori_data.sts.testlabels;
[~,train_n] = size(trainData);
fprintf('-length of train data:%d \n', train_n);
[~,test_n] = size(testData);
fprintf('-length of test data:%d \n', test_n);
a=[];b = [];

index = 1;
for i=1:train_n
    [~,wid_size] = size(trainData{1,i});
    for k = 1:wid_size
        acceleration = trainData{1,i}(1,k);
        a(index,1)=acceleration;
        Angular_velocity = trainData{1,i}(2,k);
        b(index,1)=Angular_velocity;
        index=index+1;
%         fprintf("acceleration?%f \n",acceleration);
%         fprintf("Angular_velocity?%f \n",Angular_velocity);
    end
end

acc_max = max(a);
acc_min = min(a);
ang_max = max(b);
ang_min = min(b);

newTrainData = {};
newTrainDataLabels = zeros(train_n*data_expand_scale , 1);
newTestData = {};
newTestDataLabels = zeros(test_n*data_expand_scale , 1);

index = 0;
rnd_acc_scale = unifrnd(-0.1,0.1,1,data_expand_scale);
rnd_ang_scale = unifrnd(-1,1,1,data_expand_scale);
%% generate new train data
for scale = 1:data_expand_scale
    templabel = -1;
    for i=1:train_n
        index = index+1;
        [~,wid_size] = size(trainData{1,i});
        temp_data = zeros(2,wid_size);
        templabel = trainDataLabels(i);
        for k = 1:wid_size
            acceleration = trainData{1,i}(1,k) + rnd_acc_scale(scale);
            Angular_velocity = trainData{1,i}(2,k) + rnd_ang_scale(scale);
            temp_data(1,k)=acceleration;
            temp_data(2,k)=Angular_velocity;
        end
        newTrainData{1,index} = temp_data;
        newTrainDataLabels(index) = templabel;
    end
end
%%generate new test data
index = 0;
for scale = 1:data_expand_scale
    templabel = -1;
    for i=1:test_n
        index = index+1;
        [~,wid_size] = size(testData{1,i});
        temp_data = zeros(2,wid_size);
        templabel = testDataLabels(i);
        for k = 1:wid_size
            acceleration = testData{1,i}(1,k) + rnd_acc_scale(scale);
            Angular_velocity = testData{1,i}(2,k) + rnd_ang_scale(scale);
            temp_data(1,k)=acceleration;
            temp_data(2,k)=Angular_velocity;
        end
        newTestData{1,index} = temp_data;
        newTestDataLabels(index) = templabel;
    end
end
%% generate new Mat
save_data.train = newTrainData;
save_data.trainlabels = newTrainDataLabels;
save_data.test = newTestData;
save_data.testlabels = newTestDataLabels;
save('new_sts.mat','save_data');