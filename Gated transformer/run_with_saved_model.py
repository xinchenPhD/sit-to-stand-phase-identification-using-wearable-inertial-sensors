
import torch
# print('当前使用的pytorch版本：', torch.__version__)
from utils.random_seed import setup_seed
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
from utils.heatMap import heatMap_all
from utils.TSNE import gather_by_tsne
from utils.TSNE import gather_all_by_tsne
import numpy as np
from utils.colorful_line import draw_colorful_line

setup_seed(30)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'use device: {DEVICE}')

save_model_path = r'C:\Users\Administrator\Desktop\cx\GTN-master\saved model\JapaneseVowels 97.03 batch=3.pkl'
file_name = save_model_path.split('\\')[-1].split(' ')[0]
path = f'C:\\Users\\Administrator\\Desktop\\cx\\GTN-master\\Gated Transformer 论文IJCAI版\\MTS_dataset\\{file_name}\\{file_name}.mat'
Heatmap_path = r'C:\Users\Administrator\Desktop\cx\GTN-master\Gated Transformer 论文IJCAI版\heatmap_figure_in_test'
ACCURACY = save_model_path.split('\\')[-1].split(' ')[1]

BATCH_SIZE = int(save_model_path[save_model_path.find('=')+1:save_model_path.rfind('.')])
# heatMap_or_not = False
heatMap_or_not = True
gather_or_not = False
gather_all_or_not = True

net = torch.load(save_model_path, map_location=torch.device('cuda:0'))

test_dataset = MyDataset(path, 'test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'step为最大值的sample个数:{len(test_dataset.max_length_sample_inTest)}')
if len(test_dataset.max_length_sample_inTest) == 0:
    gather_or_not = False
    heatMap_or_not = False
    print('测试集中没有step为最大值的样本， 将不能绘制make sense 的heatmap 和 gather 图 可尝试换一个数据集')

correct = 0
total = 0
with torch.no_grad():
    all_sample_X = []
    all_sample_Y = []
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre, encoding, score_input, score_channel, gather_input, gather_channel, gate = net(x.to(DEVICE), 'test')

        all_sample_X.append(encoding)
        all_sample_Y.append(y)
        if heatMap_or_not:
            for index, sample in enumerate(test_dataset.max_length_sample_inTest):
                if sample.cpu().detach().numpy().tolist() in x.cpu().detach().numpy().tolist():
                    target_index = x.cpu().detach().numpy().tolist().index(sample.cpu().detach().numpy().tolist())
                    print('正在绘制heatmap图...')
                    heatMap_all(score_input[target_index], score_channel[target_index], sample,
                                Heatmap_path, file_name, ACCURACY, index)
                    print('heatmap图绘制完成！')

        if gather_or_not:
            for index, sample in enumerate(test_dataset.max_length_sample_inTest):
                if sample.numpy().tolist() in x.numpy().tolist():
                    target_index = x.numpy().tolist().index(sample.numpy().tolist())
                    print('正在绘制gather图...')
                    gather_by_tsne(gather_input[target_index].numpy(), np.arange(gather_input[target_index].shape[0]), index, file_name+' input_gather')
                    gather_by_tsne(gather_channel[target_index].numpy(), np.arange(gather_channel[target_index].shape[0]), index, file_name+' channel_gather')
                    print('gather图绘制完成！')
                    draw_data = x[target_index].transpose(-1, -2)[0].numpy()
                    draw_colorful_line(draw_data)
                    gather_or_not = False

        _, label_index = torch.max(y_pre.data, dim=-1)
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()

    if gather_all_or_not:
        all_sample_X = torch.cat(all_sample_X, dim=0).cpu().detach().numpy()
        all_sample_Y = torch.cat(all_sample_Y, dim=0).cpu().detach().numpy()
        print('正在绘制gather图...')
        gather_all_by_tsne(all_sample_X, all_sample_Y, test_dataset.output_len, file_name+' all_sample_gather')
        print('gather图绘制完成！')

    print(f'Accuracy: %.2f %%' % (100 * correct / total))


