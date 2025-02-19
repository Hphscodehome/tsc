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
            
        self.networks2 = nn.ModuleDict()
        for i in range(self.max_length):
            self.networks2[str(i)] = nn.Sequential(
                nn.Embedding(2, 4),
                nn.Linear(4, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 7),
                nn.LeakyReLU(),
                nn.Linear(7, output_dimension),
                nn.Tanh()
            )
            
        self.networks3 = nn.ModuleDict()
        for i in range(self.max_length):
            self.networks3[str(i)] = nn.Sequential(
                nn.Linear(output_dimension, 7),
                nn.LeakyReLU(),
                nn.Linear(7, output_dimension),
                nn.Tanh()
            )
        
        self.multihead_attn = nn.MultiheadAttention(3*output_dimension, num_heads=1)
        self.weight = nn.Parameter(torch.randn(max_length))
        self.output_layer = nn.Linear(3*output_dimension, 2)
        self.apply(self._init_weights)
        
    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        batch_size, seq_length, features = x.shape
        out1 = []
        for i in range(seq_length):
            network = self.networks[str(i)]#4*16
            out1.append(network(x[:, i,-2].to(torch.int64).unsqueeze(1)).squeeze(1))
        out1 = torch.stack(out1,dim=1)
        
        out2 = []
        for i in range(seq_length):
            network = self.networks2[str(i)]#4*16
            out2.append(network(x[:, i,-1].to(torch.int64).unsqueeze(1)).squeeze(1))
        out2 = torch.stack(out2,dim=1)
        
        out3 = []
        for i in range(seq_length):
            network = self.networks3[str(i)]#4*16
            out3.append(network(x[:, i,:-2].to(torch.float32)))
        out3 = torch.stack(out3,dim=1)
        merged_tensor = torch.cat([out1, out2, out3], dim=2)
        
        query = key = value = merged_tensor.permute(1, 0, 2)  # 形状变为 (5, 4, 16)
        
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        final_output = attn_output.permute(1, 0, 2)
        
        weights = nn.Softmax(dim=-1)(nn.Tanh()(self.weight[:seq_length]))
        weights = weights.view(1,final_output.shape[1],1)  # 形状变为(1, 5, 1)
        final_output = torch.sum(final_output*weights,dim=1)
        final_output = self.output_layer(final_output)
        
        return final_output

class MyDataset(Dataset):
    def __init__(self, x_file = '/data/hupenghui/Self/tsc/ticket/data/train_x.json',y_file = '/data/hupenghui/Self/tsc/ticket/data/train_y.json',end = 9):
        with open(x_file, 'r', encoding='utf-8') as f:
            self.x = json.load(f)
        with open(y_file, 'r', encoding='utf-8') as f:
            self.y = json.load(f)
        self.x = np.array(self.x,dtype=np.float32)
        self.y = np.array(self.y,dtype=np.int64)
        self.x = self.x[:,-end:,:]
        #print(len(self.x),len(self.x[0]),self.x[0],self.y[0])
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]