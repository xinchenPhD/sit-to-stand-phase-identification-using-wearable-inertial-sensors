import torch
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm   
import os
import numpy as np
# from torchviz import make_dot
# import hiddenlayer as hl   #torch/tf/keras
# from module.encoder import Encoder
from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
# from module.feedForward import FeedForward
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'use device: {DEVICE}')
setup_seed(30)
reslut_figure_path = r'C:\Users\Administrator\Desktop\cx\GTN-master\Gated Transformer 论文IJCAI版\result_figure'

# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_feature4.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_gyr.mat'
path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_2118.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\train_test.mat'


test_interval = 5
# test_interval = 2
draw_key = 1
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]


EPOCH = 100
BATCH_SIZE = 64
LR = 0.01  # LR = 1e-4   # LR = 0.001
Momentum = 0.937
d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True
mask = True

# optimizer_name = 'Adagrad'
# optimizer_name = 'Adam'
optimizer_name = 'SGD'



train_dataset = MyDataset(path, 'train')
test_dataset = MyDataset(path, 'test')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

DATA_LEN = train_dataset.train_len
d_input = train_dataset.input_len
d_channel = train_dataset.channel_len
d_output = train_dataset.output_len


# print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')


net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)


# print(net)
# print(net.state_dict())


loss_function = Myloss()
if optimizer_name == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=Momentum, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

# if optimizer_name == 'Adagrad':
#     optimizer = optim.Adagrad(net.parameters(), lr=LR)
# elif optimizer_name == 'Adam':
#     optimizer = optim.Adam(net.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
# # elif optimizer_name == 'SGD':
# #     optimizer = optim.SGD(net.parameters(), lr=LR)


correct_on_train = []
correct_on_test = []

loss_list = []
# loss1_list = []
time_cost = 0



def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()


    for index in range(EPOCH):

        for i, (x, y) in enumerate(train_dataloader):
            # loss1 = 0
            # optimizer.zero_grad()
            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())
            # loss1 = loss.item() + loss1
            # loss1_list.append(loss1.item())
            optimizer.zero_grad()
            loss.backward()
            # loss1.backward()
            optimizer.step()
            # scheduler.step(loss)
            scheduler.step()

            # y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            # loss = loss_function(y_pre, y.to(DEVICE))
            # optimizer.zero_grad()
            # print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            # loss.backward()
            # optimizer.step()
            # loss_list.append(loss.item())
            # scheduler.step()

        # scheduler.step(np.mean(loss1))
        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
    # loss可视化
    # plt.figure()
    # plt.plot(loss_list, "-r")
    # plt.show()


    Save_Model_Path = f'C:\\Users\\Administrator\\Desktop\\cx\\GTN-master\\saved model\\{file_name} batch={BATCH_SIZE}.pkl'
    # torch.save(net.state_dict(), Save_Model_Path)
    torch.save(net, Save_Model_Path)

    pbar.update()
    Rename_Save_Model_Path = f'C:\\Users\\Administrator\\Desktop\\cx\\GTN-master\\saved model\\{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl'
    os.rename(Save_Model_Path, Rename_Save_Model_Path)

    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    end = time()
    time_cost = round((end - begin) / 60, 2)

    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)


if __name__ == '__main__':
    train()




