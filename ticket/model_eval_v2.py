import torch
from torch.distributions import Categorical
import logging,json
import argparse
from model_data_v1 import Model as Model1
from model_data_v2 import Model as Model2
from bs4 import BeautifulSoup
import logging,re,requests,argparse


def read_list_from_file(filepath):
    my_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = line.strip()  # 去除行尾的换行符
            my_list.append(item)
    return my_list

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


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    
    
    file = './data/issue_values.json'
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)#无序
    all_indexs = list(results.keys())
    all_indexs = sorted(all_indexs)# 从小到大
    
    file = './data/artificial_issues.txt'
    index = read_list_from_file(file) # 无序 部分 人工部分
    
    lanqius = get_sales() # 从大到小
    lanqius.reverse() # 从小到大 全部
    
    model1 = Model1()
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model1.load_state_dict(state_dict)
    
    # 首先构建logits数据
    logits = []
    for i,ind in enumerate(all_indexs[args.end:]):
        test = torch.tensor(lanqius[i:i+args.end]).to(torch.int64)-1
        #logging.info(f"{test}")
        with torch.no_grad():
            outputs = model1(test.unsqueeze(0))
            logits.append(outputs.squeeze(0).tolist()+[lanqius[i+args.end]-1]+[1 if ind in index else 0])
            #1 就是人工选择，0就是随机选择
            
    model = Model2()
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model2/best_modelv2_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict)
    
    test = torch.tensor(logits[len(logits)-(args.end):len(logits)-0],dtype=torch.float32)
    logging.info(f"真实测试数据是：{test}")
    with torch.no_grad():
        outputs = model(test.unsqueeze(0))
    distribution = Categorical(logits=outputs)
    samples = distribution.sample((1,))
    logging.info(f"""
输入数据是:{test},
当前模型预测结果为：
{outputs.flatten().tolist()},
{outputs.squeeze(0)},
logits转化为概率分布是:
{distribution.probs.flatten().tolist()},
logits最大的值为:
{torch.argmax(outputs.squeeze(0),dim=0)}
按照logits采样得到的结果是:
{samples.flatten()}
""")