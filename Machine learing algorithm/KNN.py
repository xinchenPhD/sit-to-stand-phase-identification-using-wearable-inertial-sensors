import time

import torch
from matplotlib import pyplot as plt
from scipy import io
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import polt

name = "matlab.mat"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")
test_data = io.loadmat(name)['sts'][0, 0]['test'].T.squeeze()
test_label = io.loadmat(name)['sts'][0, 0]['testlabels'].squeeze()
train_data = io.loadmat(name)['sts'][0, 0]['train'].T.squeeze()
train_label = io.loadmat(name)['sts'][0, 0]['trainlabels'].squeeze()

train_len = train_data.shape[0]
test_len = test_data.shape[0]
output_len = len(tuple(set(train_label)))

max_lenth = 0  # 93
for item in train_data:
    item = torch.as_tensor(item).float()
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]
        # max_length_index = train_data.tolist().index(item.tolist())

for item in test_data:
    item = torch.as_tensor(item).float()
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]

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
train_dataset = torch.flatten(torch.stack(train_dataset, dim=0).permute(0, 2, 1), start_dim=1)
test_dataset = torch.flatten(torch.stack(test_dataset, dim=0).permute(0, 2, 1), start_dim=1)
train_label = torch.tensor(train_label, dtype=torch.long) - 1
test_label = torch.tensor(test_label, dtype=torch.long) - 1

print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(train_dataset.shape, train_label.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(test_dataset.shape, test_label.shape))


train_data = train_dataset.numpy()
train_label = train_label.numpy()
test_data = test_dataset.numpy()
test_label = test_label.numpy()
time_start = time.time()
# create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # 96

# fit the classifier to the data
knn.fit(train_data, train_label)

time_end = time.time()
time_sum = time_end - time_start
# predict on the test set
y_pred = knn.predict(test_data)
# accuracy = accuracy_score(test_label, y_pred)
confusion = confusion_matrix(test_label, y_pred)
polt.plot_matrix(test_label, y_pred, 'KNN-HOT')
print(y_pred, test_label)
# print(accuracy)
print("Confusion Matrix:\n", confusion)
print(classification_report(test_label, y_pred))
print(time_sum)


