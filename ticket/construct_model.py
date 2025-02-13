import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
import json
import numpy as np
import logging
import argparse

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
        self.x = np.array(self.x,dtype=int)
        self.y = np.array(self.y,dtype=int)
        self.x = self.x[:,-end:]
        self.x -= 1
        self.y -= 1
        #print(len(self.x),len(self.x[0]),self.x[0],self.y[0])
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=7)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    dataset = MyDataset(end = args.end)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_size//3+1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_size//2+1, shuffle=False)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100000
    
    best_val_loss = float('inf')  # 初始化最佳验证集损失
    patience = 15  # 设置早停的耐心值
    counter = 0  # 计数器，记录连续多少个epoch验证集损失没有下降
    
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data.clone().detach().to(torch.int)
            labels = labels.clone().detach().to(torch.long) # unsqueeze(1) 增加一个维度，使其形状为 (batch_size, 1)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 3 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        # 验证循环
        val_loss = 0.0
        with torch.no_grad():  # 在验证阶段不需要计算梯度
            for data, labels in val_loader:
                data = data.to(torch.int)  # 验证集数据也需要转换类型
                labels = labels.to(torch.long)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_val_loss))
        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # 重置计数器
            # 保存模型
            torch.save(model.state_dict(), f'best_model_{args.end}.pth') # 保存当前最好的模型
            logging.info("Model saved!")
        else:
            counter += 1
            if counter >= patience:
                logging.info('Early stopping!')
                break  # 提前结束训练循环
    logging.info(f"当前试验截断值为：{args.end},文件名为：best_model_{args.end}.pth,最优模型验证损失为：{best_val_loss}")
    with open('record.txt', 'a', encoding='utf-8') as file:
        file.write(f"当前试验截断值为：{args.end},文件名为：best_model_{args.end}.pth,最优模型验证损失为：{best_val_loss}\n")
        #logging.info(f"{list(dataset.x[0][1:])+[int(dataset.y[0])]}")
        #logging.info(f"{torch.from_numpy(np.array(list(dataset.x[0][1:])+[int(dataset.y[0])], dtype=np.int64)).to(torch.int).unsqueeze(0)}")
        with torch.no_grad():
            outputs = model(torch.tensor([14,3,8,5,10,10,5,7,10,10,13,4][-args.end:]).to(torch.int).unsqueeze(0))
            #outputs = model(torch.from_numpy(np.array(list(dataset.x[0][1:])+[int(dataset.y[0])], dtype=np.int64)).to(torch.int).unsqueeze(0))
        file.write(f"INput:{torch.from_numpy(np.array(list(dataset.x[0][1:])+[int(dataset.y[0])], dtype=np.int64)).to(torch.int).unsqueeze(0)}, \
                   当前模型预测结果为：{outputs}, \
                   估计值为：{torch.argmax(outputs,dim=1)+1}\n")