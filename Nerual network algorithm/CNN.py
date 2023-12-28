import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy import io
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

import polt

name = "new_sts_22600.mat"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load data
print("Loading data...")
test_data = io.loadmat(name)['sts'][0, 0]['test'].T.squeeze()
test_label = io.loadmat(name)['sts'][0, 0]['testlabels'].squeeze()
train_data = io.loadmat(name)['sts'][0, 0]['train'].T.squeeze()
train_label = io.loadmat(name)['sts'][0, 0]['trainlabels'].squeeze()

train_len = train_data.shape[0]
test_len = test_data.shape[0]
output_len = len(tuple(set(train_label)))

# timestep max
max_lenth = 0
for item in train_data:
    item = torch.as_tensor(item).float()
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]
        # max_length_index = train_data.tolist().index(item.tolist())

for item in test_data:
    item = torch.as_tensor(item).float()
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]

# Padding 0
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

train_data = Data.TensorDataset(train_dataset, train_label)
test_data = Data.TensorDataset(test_dataset, test_label)
train_dataloader = Data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = Data.DataLoader(test_data, batch_size=32, shuffle=False)

input_channels = max_lenth
hidden_size = 256
input_size = 9

num_layers = 1
num_classes = 5
learning_rate = 1e-4
epochs = 100


class CNN(nn.Module):
    def __init__(self, input_channels, hidden_size, input_size, output_dim):    #   input  第一个维度为batch 第二个维度为序列长度   第三个维度是通道数
        super(CNN, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # input dim [2, 113, 2]
        self.con1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size,  input_size),
        )
        # self.con2 = nn.Sequential(
        #     nn.Conv1d(64, 128, 2, 1, 1), # [128, 1, 256]
        #     nn.ReLU(),
        # )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        out = self.con1(x)  # [2, 256, 1]
        # out = self.con2(out)
        out = out.view(out.size()[0], -1)  # [2, 256, 1] ——> [2, 256]
        out = self.fc(out)
        return out


model = CNN(input_channels, hidden_size, input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_loss = []
train_acc = []
test_loss = []
test_acc = []
x = []
total_step = len(train_dataloader)
for epoch in range(epochs):
    sum_loss = 0
    train_correct = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        # print('size', images.shape)
        # Forward pass
        outputs = model(images, device)
        # print(outputs.size())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        sum_loss += loss.data.cpu()
        train_correct += (predicted == labels).cpu().sum().item()

        # if (i + 1) % 2 == 0:
    print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss /len(test_dataloader)))
    print('Test Accuracy of train : {} %'.format(100 * train_correct / train_len))
    train_loss.append(sum_loss / len(test_dataloader))
    c1 = train_correct / train_len
    train_acc.append(train_correct / train_len)
    x.append(epoch + 1)

my_x_ticks = np.arange(1, epochs, 5)
plt.figure()
plt.plot(x, train_loss, label='Loss')
plt.legend()
plt.savefig('CNN-loss.png')
plt.figure()
plt.plot(x, train_acc, label='accuracy')
plt.xticks(my_x_ticks)
plt.legend()
plt.savefig('CNN-acc.png')
plt.show()

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    y_ture = []
    for images, labels in test_dataloader:
        labels = labels.to(device)
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
polt.plot_matrix(y_ture, y_pred, 'CNN-HOT')
print("Confusion Matrix:\n", confusion)
# Save the model checkpoint
torch.save(model.state_dict(), 'CNN-model.ckpt')
