import json
from bs4 import BeautifulSoup
import logging,re,requests,argparse
import numpy as np
from model_data_v2 import Model
import torch

def get_sales():
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=00000&end=99999'#"http://datachart.500.com/ssq/history/newinc/history.php?start=00001&end=2024097"#'https://datachart.500.com/ssq/history/outball.shtml'
    response = requests.get(url)
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    results = []
    for row in table.find_all('tr', class_='t_tr1'):  # 使用 class_ 过滤
        cells = row.find_all('td')
        results.append(int(cells[7].text.strip()))
    return results

def read_list_from_file(filepath):
    my_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = line.strip()  # 去除行尾的换行符
            my_list.append(item)
    return my_list

# 形状为前100期logits及对应的label，预测当前一期的label

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    
    file = './data/issue_values.json'
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f) # 无序 全部
    all_indexs = list(results.keys())
    all_indexs = sorted(all_indexs) # 从小到大
    
    lanqius = get_sales() # 从大到小
    lanqius.reverse() # 从小到大 全部
    
    file = './data/artificial_issues.txt'
    index = read_list_from_file(file) # 无序 部分 人工部分
    
    # 首先构建训练logits模型的数据
    train_data_x = []
    train_data_y = []
    
    for i,ind in enumerate(all_indexs[args.end:]):
        if ind in index:
            train_data_x.append(lanqius[i:i+args.end])
            train_data_y.append(lanqius[i+args.end])
            
    with open("./data/train_x_5.json", "w", encoding='utf-8') as f:
        json.dump(train_data_x, f, indent=4)
    with open("./data/train_y_5.json", "w", encoding='utf-8') as f:
        json.dump(train_data_y, f, indent=4)
        
        
    model = Model()
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict)
    
    # 首先构建logits数据
    logits = []
    for i,ind in enumerate(all_indexs[args.end:]):
        test = torch.tensor(lanqius[i:i+args.end]).to(torch.int64)
        with torch.no_grad():
            outputs = model(test.unsqueeze(0))
            logits.append(outputs.squeeze(0).tolist()+[lanqius[i+args.end]-1]+[1 if ind in index else 0])
            # logits，蓝球，是否人工设计
    
    # 然后构建训练数据
    train_x = []
    train_y = []
    for i in range(len(logits)-args.end):
        train_x.append(logits[i:i+args.end])
        train_y.append(logits[i+args.end][-1])
    with open("./data/train_x.json", "w", encoding='utf-8') as f:
        json.dump(train_x, f, indent=4)
    with open("./data/train_y.json", "w", encoding='utf-8') as f:
        json.dump(train_y, f, indent=4)
    
    print('done')
    