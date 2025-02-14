import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
import json
import numpy as np
import logging
import argparse
from model_data_v1 import Model

'''
class Model(nn.Module):
    def __init__(self,output_dimension=16):
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
        
    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self,x):
        x = self.network(x)
        x = torch.mean(x,dim=1)
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
'''
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    model = Model()
    test = torch.tensor([1,3,15,10,14,3,8,5,10,10,5,7,10,10,13,4,14][-args.end:]).to(torch.int64).unsqueeze(0)
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model/best_model_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict)
    #for name, param in model.named_parameters():
    #    print(f"参数名称: {name}, 参数形状: {param.shape}, 参数值：{param}")
    with torch.no_grad():
        outputs = model(test)
        #outputs = model(torch.from_numpy(np.array(list(dataset.x[0][1:])+[int(dataset.y[0])], dtype=np.int64)).to(torch.int).unsqueeze(0))
    logging.info(f"INput:{test.flatten().tolist()},\n当前模型预测结果为：{outputs.flatten().tolist()},\n估计值为：{torch.argmax(outputs,dim=1)+1}\n")