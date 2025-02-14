import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import logging
import argparse
from model_data_v2 import Model,MyDataset

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    
    dataset = MyDataset(end = args.end)
    train_size = int(0.81 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_size//3+1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
    logging.info(f"train groups：{len(train_loader)},eval groups：{len(val_loader)}")
    
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100000
    best_val_loss = float('inf')  # 初始化最佳验证集损失
    patience = 20  # 设置早停的耐心值
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
        
        val_loss = 0.0
        with torch.no_grad():  # 在验证阶段不需要计算梯度
            for data, labels in val_loader:
                data = data.clone().detach().to(torch.int)  # 验证集数据也需要转换类型
                labels = labels.clone().detach().to(torch.long)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # 重置计数器
            torch.save(model.state_dict(), f'/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth') # 保存当前最好的模型
            logging.info("Model saved!")
        else:
            counter += 1
            if counter >= patience:
                logging.info('Early stopping!')
                break  # 提前结束训练循环
    logging.info(f"当前试验截断值为：{args.end},文件名为：/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth,最优模型验证损失为：{best_val_loss}")
    with open('/data/hupenghui/Self/tsc/ticket/record.txt', 'a', encoding='utf-8') as file:
        file.write(f"当前试验截断值为：{args.end},文件名为：/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth,最优模型验证损失为：{best_val_loss}\n")