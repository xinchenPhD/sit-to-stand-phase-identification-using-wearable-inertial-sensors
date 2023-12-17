import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy import io
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, classification_report
from torch.autograd import Variable

import polt

# 加载数据
print("Loading data...")
name = "new_sts_22600.mat"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据
print("Loading data...")
test_data = io.loadmat(name)['sts'][0, 0]['test'].T.squeeze()  # 第一个[0, 0]一定要，否则会报错
test_label = io.loadmat(name)['sts'][0, 0]['testlabels'].squeeze()
train_data = io.loadmat(name)['sts'][0, 0]['train'].T.squeeze()
train_label = io.loadmat(name)['sts'][0, 0]['trainlabels'].squeeze()

train_len = train_data.shape[0]
test_len = test_data.shape[0]
output_len = len(tuple(set(train_label)))

# 时间步最大值
max_lenth = 0  # 93
for item in train_data:
    item = torch.as_tensor(item).float()  # torch.as_tensor或者torch.from_numpy将numpy数组转为张量
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]
        # max_length_index = train_data.tolist().index(item.tolist())

for item in test_data:
    item = torch.as_tensor(item).float()
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]

# 填充Padding  使用0进行填充
# train_data, test_data为numpy.object 类型，不能直接对里面的numpy.ndarray进行处理
train_dataset_with_no_paddding = []
test_dataset_with_no_paddding = []
train_dataset = []
test_dataset = []
max_length_sample_inTest = []
for x1 in train_data:
    train_dataset_with_no_paddding.append(x1.transpose(-1, -2).tolist())
    x1 = torch.as_tensor(x1).float()
    if x1.shape[1] != max_lenth:
        padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
        x1 = torch.cat((x1, padding), dim=1)
    train_dataset.append(x1)

for index, x2 in enumerate(test_data):
    test_dataset_with_no_paddding.append(x2.transpose(-1, -2).tolist())
    x2 = torch.as_tensor(x2).float()
    if x2.shape[1] != max_lenth:
        padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
        x2 = torch.cat((x2, padding), dim=1)
    else:
        max_length_sample_inTest.append(x2.transpose(-1, -2))
    test_dataset.append(x2)

# 最后维度 [数据条数,时间步数最大值,时间序列维度]
# train_dataset_with_no_paddding = torch.stack(train_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
# test_dataset_with_no_paddding = torch.stack(test_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
train_label = torch.tensor(train_label, dtype=torch.long) - 1
test_label = torch.tensor(test_label, dtype=torch.long) - 1
channel = test_dataset[0].shape[-1]
input = test_dataset[0].shape[-2]

# Min-Max scaling
min_a = torch.min(train_dataset)
max_a = torch.max(train_dataset)
train_dataset = (train_dataset - min_a) / (max_a - min_a)
# print(train_dataset, torch.mean(train_dataset), torch.std(train_dataset))

min_a = torch.min(test_dataset)
max_a = torch.max(test_dataset)
test_dataset = (test_dataset - min_a) / (max_a - min_a)
# print(test_dataset, torch.mean(test_dataset), torch.std(test_dataset))

# y_train = data_trainlabels - 1
# y_test = data_testlabels - 1
# y_train = np_utils.to_categorical(y_train)  # one-hot encoding
# y_test = np_utils.to_categorical(y_test)  # one-hot encoding

print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(train_dataset.shape, train_label.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(test_dataset.shape, test_label.shape))
# 将转化维张量的数据跟标签放到一起
train_data = Data.TensorDataset(train_dataset, train_label)
test_data = Data.TensorDataset(test_dataset,test_label)
train_dataloader = Data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = Data.DataLoader(test_data, batch_size=32, shuffle=False)



num_i = max_lenth * 2
num_h = 256
num_o = 5
learning_rate = 1e-4
epochs = 100

# class MLPmodel(nn.Module):
#       def __init__(self, num_i, num_h, num_o):
#             super(MLPmodel, self).__init__()
#             # 定义第一个隐藏层
#             self.hidden1 = nn.Linear(num_i, num_h)
#             self.active1 = nn.ReLU()
#             # 定义第二个隐藏层
#             self.hidden2 = nn.Linear(num_h, num_h)
#             self.active2 = nn.ReLU()
#             #定义预测回归
#             self.regression = nn.Linear(num_h, num_o)
#       def forward(self, x):
#             x = self.hidden1(x)
#             x = self.active1(x)
#             x = self.hidden2(x)
#             x = self.active2(x)
#             output = self.regression(x)
#
#             # y = F.sigmoid(self.hidden1(x))
#             # y = F.softmax(self.hidden2(y), dim=1)
#
#             return output

class MLPmodel(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLPmodel, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(num_i, num_h)
        self.active1 = nn.ReLU()
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(num_h, num_h)
        self.active2 = nn.ReLU()
        self.classification = nn.Sequential(
            nn.Linear(num_h, num_o),
            nn.Sigmoid()
        )
        # self.classification = nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.classification(x)
        return output


model = MLPmodel(num_i, num_h, num_o).to(device)
print(model)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),  lr=learning_rate)


train_loss = []
train_acc = []
test_loss = []
test_acc = []
x = []
time_start = time.time()
for epoch in range(epochs):
    sum_loss = 0
    train_correct = 0
    for data in train_dataloader:
        inputs, labels = data  # inputs 维度：[2,113,2]
        #     print(inputs.shape)
        inputs = torch.flatten(inputs, start_dim=1) # 展平数据，转化为[2,226]
        #     print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        sum_loss += loss.data.cpu()
        train_correct += (predicted == labels).cpu().sum().item()

    print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(train_dataloader)))
    print('        correct:%.03f%%' % (100 * train_correct / train_len))
    train_loss.append(sum_loss / len(train_dataloader))
    train_acc.append(train_correct / train_len)
    x.append(epoch + 1)

time_end = time.time()
time_sum = time_end - time_start
my_x_ticks = np.arange(1, epochs, 5)
plt.figure()
plt.plot(x, train_loss, label='Loss')
plt.legend()
plt.savefig('MLP-20-loss.png')
plt.figure()
plt.plot(x, train_acc, label='accuracy')
plt.xticks(my_x_ticks)
plt.legend()
plt.savefig('MLP-20-acc.png')
plt.show()

# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    y_ture = []
    for images, labels in test_dataloader:
        labels = labels.to(device)
        images = torch.flatten(images, start_dim=1)
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pred = (np.concatenate((y_pred, predicted.cpu().numpy()), axis=0))
        y_ture = (np.concatenate((y_ture, labels.cpu().numpy()), axis=0))
    y_pred = y_pred.astype(np.uint8)
    y_ture = y_ture.astype(np.uint8)
    print('Test Accuracy of test : {} %'.format(100 * correct / total))

confusion = confusion_matrix(y_ture, y_pred)
polt.plot_matrix(y_ture, y_pred, 'MLP-20-HOT')
print("Confusion Matrix:\n", confusion)
# Save the model checkpoint
torch.save(model.state_dict(), 'MLP-20-model.ckpt')
print(classification_report(y_ture, y_pred))
print(time_sum)
