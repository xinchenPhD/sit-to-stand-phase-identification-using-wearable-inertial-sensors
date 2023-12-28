import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTM_norm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTM_norm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # x.size(0)是batch_size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTM_Stacked(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM_Stacked, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers - 1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers - 1, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out

class LSTM_Bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM_Bi, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM_CNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, device):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_DeepConv(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM_DeepConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, device):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# class LSTM_CONV(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
#         super(LSTM_CONV, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=(3, input_size), padding=1)
#         self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_size * 2, output_size)
#     def forward(self, x, device):
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = self.pool(x)
#         x = x.squeeze(2)
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out)
#         out = self.fc(out[:, -1, :])
#         return out

class LSTM_ENCODER_DECODER(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM_ENCODER_DECODER, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, device):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# our proposed algorithm
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1)  #
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)  # BiLSTM
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.attention = nn.Linear(hidden_dim*2, 1)

    def forward(self, x, device):
        x = x.permute(0, 2, 1)    #
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        x, _ = self.lstm(x, (h0, c0))

        attn_weights = nn.functional.softmax(self.attention(x), dim=1)
        attn_out = torch.bmm(attn_weights.permute(0, 2, 1), x).squeeze().to(device)

        out = self.fc(attn_out)
        return out



