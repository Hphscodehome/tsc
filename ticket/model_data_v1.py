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
                nn.Linear(7, output_dimension)
            )
        self.multihead_attn = nn.MultiheadAttention(output_dimension, num_heads=1)
        self.weight = nn.Parameter(torch.randn(max_length))
        self.apply(self._init_weights)
        
    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        batch_size, seq_length = x.shape
        out = []
        for i in range(seq_length):
            network = self.networks[str(i)]#4*16
            out.append(torch.clamp(network(x[:, i].unsqueeze(1)),min=-5,max=2).squeeze(1))
        out = torch.stack(out,dim=1)
        query = key = value = out.permute(1, 0, 2)  # 形状变为 (5, 4, 16)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        final_output = attn_output.permute(1, 0, 2)
        weights = nn.Softmax(dim=-1)(torch.clamp(self.weight[:seq_length],min=-5,max=2))
        weights = weights.view(1,final_output.shape[1],1)  # 形状变为(1, 5, 1)
        final_output = torch.sum(final_output*weights,dim=1)
        return final_output

class MyDataset(Dataset):
    def __init__(self, x_file = '/data/hupenghui/Self/tsc/ticket/data/train_x_5.json',y_file = '/data/hupenghui/Self/tsc/ticket/data/train_y_5.json',end =7):
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