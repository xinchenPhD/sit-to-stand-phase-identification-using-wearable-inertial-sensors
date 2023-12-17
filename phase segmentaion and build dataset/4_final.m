load("test.mat");
load("trainlabels.mat");
load("train.mat");
load("testlabels.mat");
sts.test=test;
sts.trainlabels=trainlabels;
sts.train=train;
sts.testlabels=testlabels;
save('sts_1758.mat',"sts");







