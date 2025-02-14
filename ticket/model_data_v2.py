import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
# 简单版本

class Model(nn.Module):
    def __init__(self, output_dimension=16, max_length=51):
        super(Model, self).__init__()
        self.max_length = max_length
        self.networks = nn.ModuleDict()
        for i in range(self.max_length):
            self.networks[str(i)] = nn.Sequential(
                nn.Embedding(output_dimension, 4),
                nn.Linear(4, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 7),
                nn.LeakyReLU(),
                nn.Linear(7, output_dimension),
                nn.Tanh()
            )
        self.apply(self._init_weights)
        
    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        batch_size, seq_length = x.shape
        out = torch.zeros(batch_size, self.networks[str(0)][0].num_embeddings).to(x.device)  # 使用第一个网络的embedding_dim确定输出维度
        for i in range(seq_length):
            network = self.networks[str(i)]
            out += network(x[:, i].unsqueeze(1)).squeeze(1)
        return out

class MyDataset(Dataset):
    def __init__(self, x_file = '/data/hupenghui/Self/tsc/ticket/data/train_x.json',y_file = '/data/hupenghui/Self/tsc/ticket/data/train_y.json',end =7):
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