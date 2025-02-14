import json
from bs4 import BeautifulSoup
import logging,re,requests
import numpy as np

def get_sales():
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=09011&end=99999'#"http://datachart.500.com/ssq/history/newinc/history.php?start=00001&end=2024097"#'https://datachart.500.com/ssq/history/outball.shtml'
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
    file = '/data/hupenghui/Self/tsc/ticket/data/issue_values.json'
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)#无序
    
    file = '/data/hupenghui/Self/tsc/ticket/data/artificial_issues.txt'
    index = read_list_from_file(file)#无序

    lanqius = get_sales()#从大到小
    lanqius.reverse()# 从小到大
    
    with open("/data/hupenghui/Self/tsc/ticket/data/blue_records.json", "w", encoding='utf-8') as f:
        json.dump(lanqius, f, indent=4)
        
    all_indexs = list(results.keys())
    all_indexs = sorted(all_indexs)# 从小到大
    
    train_data_x = []
    train_data_y = []
    for ind in index:
        i = all_indexs.index(ind)
        if lanqius[i+100] == int(results[ind]['blue'].strip()):
            train_data_x.append(lanqius[i:i+100])
            train_data_y.append(int(results[ind]['blue'].strip()))
        else:
            print(i,ind)
            train_data_x.append(lanqius[i:i+100])
            train_data_y.append(lanqius[i+100])
    with open("/data/hupenghui/Self/tsc/ticket/data/train_x.json", "w", encoding='utf-8') as f:
        json.dump(train_data_x, f, indent=4)
    with open("/data/hupenghui/Self/tsc/ticket/data/train_y.json", "w", encoding='utf-8') as f:
        json.dump(train_data_y, f, indent=4)
    print('done')
    