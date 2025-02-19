import json
from bs4 import BeautifulSoup
import logging,re,requests,argparse


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
    assert len(lanqius) == len(all_indexs),"invalid length"
    
    file = './data/artificial_issues.txt'
    index = read_list_from_file(file) # 无序 部分 人工部分
    
    # 首先构建训练logits模型的数据
    train_data_x = []
    train_data_y = []
    
    for i,ind in enumerate(all_indexs[args.end:]):
        #print(ind,type(ind),index[0],type(index[0]))
        if (ind in index):
            if (results[ind]['blue'] == lanqius[i+args.end]):
                train_data_x.append(lanqius[i:i+args.end])
                train_data_y.append(lanqius[i+args.end])
            else:
                print(f"error{ind}")
                break
            
    with open("./data/train_x_5.json", "w", encoding='utf-8') as f:
        json.dump(train_data_x, f, indent=4)
    with open("./data/train_y_5.json", "w", encoding='utf-8') as f:
        json.dump(train_data_y, f, indent=4)
    
    print('done')
    