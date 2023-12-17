# 基于pytorch框架，利用Gated Transformer网络模型用于多变量时间序列分类
import torch
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm   # 进度条
import os
import numpy as np
# from torchviz import make_dot   # 网络模型结构可视化，可以指定输入值和预测值
# import hiddenlayer as hl   #支持torch/tf/keras
# from module.encoder import Encoder
from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
# from module.feedForward import FeedForward
import matplotlib.pyplot as plt
# 1、设置GPU索引
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'use device: {DEVICE}')
setup_seed(30)  # 设置随机数初始化种子
reslut_figure_path = r'C:\Users\Administrator\Desktop\cx\GTN-master\Gated Transformer 论文IJCAI版\result_figure'  # 结果图像保存路径

# 数据集路径选择 数据量尽量多、数据集划分
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_feature4.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_gyr.mat'
path = r'C:\Users\Administrator\Desktop\cx\相位划分\sts_2118.mat'
# path = r'C:\Users\Administrator\Desktop\cx\相位划分\train_test.mat'


test_interval = 5  # 测试间隔 单位：epoch
# test_interval = 2
draw_key = 1  # 大于等于draw_key才会保存图像
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # 获得文件名字


# 超参数调整
EPOCH = 100  # 训练迭代次数,轮次数    #批量大小选取越大，迭代次数也要相应增加
BATCH_SIZE = 64   # 每轮迭代采样X个样本组成小批量  # 随机梯度下降 [32-256]
LR = 0.01  # LR = 1e-4   # LR = 0.001
Momentum = 0.937  # 不需要动态   [0,1)  #0.937
d_model = 512      # 词嵌入维度，保证模块衔接维度相同
d_hidden = 1024   # feedForward模块中隐藏层的维度
q = 8
v = 8
h = 8
N = 8
dropout = 0.2  # 防止过拟合
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask

# 优化器选择
# optimizer_name = 'Adagrad'
# optimizer_name = 'Adam'
optimizer_name = 'SGD'


# 训练集和测试集在同一个路径下
train_dataset = MyDataset(path, 'train')
test_dataset = MyDataset(path, 'test')
# 定义一个数据加载器，数据集批量处理
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)   # shuffle=True每次迭代前打乱数据
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

DATA_LEN = train_dataset.train_len  # 训练集样本数量
d_input = train_dataset.input_len  # 时间步数量
d_channel = train_dataset.channel_len  # 时间序列维度
d_output = train_dataset.output_len  # 分类类别

# 维度展示
# print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# 创建Gated-Transformer模型
# 网络结构太复杂，导致过拟合，数据量大用层数多的模型
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
# 2、网络转换为cuda模式： .to(device)

# print(net)   # 不直观
# print(net.state_dict())
# 创建其他模型类


# 创建loss函数 分类任务使用交叉熵损失
loss_function = Myloss()
# 定义优化器
if optimizer_name == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=Momentum, weight_decay=1e-2)   # 加入L2正则化防止过拟合，可以降低网络复杂度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)  # 指数式调整
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

# if optimizer_name == 'Adagrad':
#     optimizer = optim.Adagrad(net.parameters(), lr=LR)
# elif optimizer_name == 'Adam':
#     optimizer = optim.Adam(net.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)  # 指数式调整
# # elif optimizer_name == 'SGD':
# #     optimizer = optim.SGD(net.parameters(), lr=LR)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
# loss1_list = []
time_cost = 0


# 测试函数
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)    # 3、 数据转换为cuda模式：data.to(Device)
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


# 训练模型
def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()

    # 进行训练，输出每次迭代的损失函数
    for index in range(EPOCH):
        # 训练的加载器迭代计算
        for i, (x, y) in enumerate(train_dataloader):
            # loss1 = 0
            # optimizer.zero_grad()
            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())
            # loss1 = loss.item() + loss1
            # loss1_list.append(loss1.item())
            optimizer.zero_grad()   # 优化器梯度清零
            loss.backward()   # 损失的反向传播，计算梯度
            # loss1.backward()
            optimizer.step()   # 参数更新优化
            # scheduler.step(loss)
            scheduler.step()

            # y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            # loss = loss_function(y_pre, y.to(DEVICE))
            # optimizer.zero_grad()
            # print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            # loss.backward()  # 损失的后向传播，计算梯度
            # optimizer.step()  # 使用梯度进行优化
            # loss_list.append(loss.item())
            # scheduler.step()

        # scheduler.step(np.mean(loss1))
        # 提前停止训练，保存效果最好的模型参数
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

    # 训练好的模型路径保存
    Save_Model_Path = f'C:\\Users\\Administrator\\Desktop\\cx\\GTN-master\\saved model\\{file_name} batch={BATCH_SIZE}.pkl'
    # 训练好的模型保存
    # torch.save(net.state_dict(), Save_Model_Path)      #只保存网络的参数
    torch.save(net, Save_Model_Path)   # 整个网络

    pbar.update()
    Rename_Save_Model_Path = f'C:\\Users\\Administrator\\Desktop\\cx\\GTN-master\\saved model\\{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl'
    os.rename(Save_Model_Path, Rename_Save_Model_Path)

    # 查看训练好的模型参数
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # 绘制loss和accuracy曲线图
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)


if __name__ == '__main__':
    train()




