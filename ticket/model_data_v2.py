import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
import json
import numpy as np
# 简单版本

class Model(nn.Module):
    def __init__(self, output_dimension=16, max_length=100):
        super(Model,self).__init__()
        self.network = nn.Sequential(
                    nn.Embedding(output_dimension, 4),
                    nn.Linear(4, 5),
                    nn.LeakyReLU(),
                    nn.Linear(5, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, output_dimension),
                    nn.Tanh()
                )
        self.apply(self._init_weights)
        self.position_weights = nn.Parameter(torch.randn(max_length))
        self.position_weights_tanh = nn.Tanh()
        self.position_weights_softmax = nn.Softmax(dim=1)
        
    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self,x):
        batch_size, seq_length = x.shape
        pos_weights = self.position_weights[:seq_length].unsqueeze(0).expand(batch_size, -1)
        x_emb = self.network[0](x)  # 获取嵌入层输出
        x_feat = self.network[1:](x_emb)  # 应用剩余的网络层
        weighted_x = x_feat * self.position_weights_softmax(self.position_weights_tanh(pos_weights)).unsqueeze(-1)  # (batch, length, output_dimension)
        x = torch.sum(weighted_x, dim=1)
        return x

class MyDataset(Dataset):
    def __init__(self, x_file = '/data/hupenghui/Self/tsc/ticket/train_x.json',y_file = '/data/hupenghui/Self/tsc/ticket/train_y.json',end =7):
        with open(x_file, 'r', encoding='utf-8') as f:
            self.x = json.load(f)
        with open(y_file, 'r', encoding='utf-8') as f:
            self.y = json.load(f)
        self.x = np.array(self.x,dtype=np.int64)
        self.y = np.array(self.y,dtype=np.int64)
        self.x = self.x[:,-end:]
        self.x -= 1
        self.y -= 1
        #print(len(self.x),len(self.x[0]),self.x[0],self.y[0])
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]