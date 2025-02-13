import requests
import re
from bs4 import BeautifulSoup
import logging
from collections import defaultdict
import numpy as np
import json

#Dict[index,Dict[attr,value]]
#attr in [red,blue,6,5,4,3,2,1,sales]
index_values = defaultdict(lambda : {})
def get(url, headers):
    global index_values
    response = requests.get(url, headers = headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        numbers = soup.select('.winning-results .number i')
        sales = soup.select_one('.article .gray2:nth-of-type(1)')
        sales_number = sales.text.replace(',', '').strip()
        sales_number = re.search(r'\d+', sales_number).group()
        if numbers and (int(sales_number) != 0):
            match = re.search(r'/(\d+)\.html$', url)
            number = match.group(1)
            number = int(number)
            #logging.info(f"本期号:{number}")
            red_numbers = []
            blue_number = ""
            for num in numbers:
                if 'c-red' in num['class']:
                    red_numbers.append(num.text)
                else:  # 假设蓝色号码没有class为c-red的i元素
                    blue_number = num.text
            index_values[number]['red'] = red_numbers
            index_values[number]['blue'] = blue_number
            #logging.info(f"红球号码:{red_numbers}")
            #logging.info(f"蓝球号码:{blue_number}")
            # 提取中奖信息
            winners = soup.select('.winner-list tbody tr')
            for winner in winners:
                cells = winner.find_all('td')
                if len(cells) == 3:
                    award_name = cells[0].text.strip()
                    count = cells[1].text.strip()
                    amount = cells[2].text.strip()
                    index_values[number][award_name[:3]] = int(count)
                    #logging.info(f"{award_name}: {count} 注，每注 {amount} 元")
            '''
            # 提取销量、奖池和截止时间
            sales = soup.select_one('.article .gray2:nth-of-type(1)')
            pool = soup.select_one('.article .yellow')
            deadline = soup.select_one('.article .gray2:nth-of-type(2)')
            if sales:
                #logging.info(f"本期销量:{sales.text}")
                sales_number = sales.text.replace(',', '').strip()
                sales_number = re.search(r'\d+', sales_number).group()
                index_values[number]['sales'] = int(sales_number)
            if pool:
                True
                #logging.info(f"奖池滚存:{pool.text}")
            if deadline:
                True
                #logging.info(f"兑奖截止时间:{deadline.text}")
            '''
            return True
        else:
            return False
    else:
        #logging.info(f"请求失败，状态码：{response.status_code}")
        return False

def get_sales():
    #global index_values
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=24102&end=24108'#"http://datachart.500.com/ssq/history/newinc/history.php?start=00001&end=2024097"#'https://datachart.500.com/ssq/history/outball.shtml'
    response = requests.get(url)
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    header = [th.text for th in table.find_all('th')]
    print("表头:", header)
    data = []
    # 遍历表格的每一行（除了表头行）
    for row in table.find_all('tr', class_='t_tr1'):  # 使用 class_ 过滤
        row_data = []
        for cell in row.find_all('td'):
            # 清理单元格文本，移除多余空白
            cleaned_text = cell.text.strip().replace('\xa0', '') # Remove non-breaking spaces
            row_data.append(cleaned_text)
        data.append(row_data)
    print("数据:")
    for row in data:
        print(row)
def process(results):
    _keys = list(results.keys())
    _keys = sorted(_keys,reverse=True)
    tongjis = []
    qihaos = []
    for key in _keys:
        # 回收
        huishou = 0
        # 售出
        shouchu = results[key]['sales']/2
        for index_key in results[key].keys():
            if index_key.endswith('等奖'):
                huishou += results[key][index_key]
        # 占比
        tongjis.append(huishou/shouchu)
        qihaos.append(key)
    logging.info(tongjis)
    tongjis = np.array(tongjis)
    qihaos = np.array(qihaos)
    bigresults = tongjis[tongjis > 1/16]
    smallresults = tongjis[tongjis <= 1/16]
    bigqihaos = qihaos[tongjis > 1/16]
    smallqihaos = qihaos[tongjis <= 1/16]
    logging.info(f"""大于1/16的占比有：{len(bigresults)/len(tongjis)}
                 小于1/16的占比有：{len(smallresults)/len(tongjis)}
                 大于1/16的期号是：{bigqihaos}
                 小于1/16的期号是：{smallqihaos}
                 """)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    '''
    for year in range(25,8,-1):
        for index in range(160,0,-1):
            url = f"https://m.78500.cn/kaijiang/ssq/2{str(year).zfill(3)+str(index).zfill(3)}.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
            }
            if get(url, headers):
                True
                #break
    logging.info(f"{index_values}")
    with open('dd_index_values.json', 'w') as f:
        json.dump(index_values, f, indent=4)  # indent=4 表示缩进 4 个空格
    process(index_values)
    
    url = "https://m.78500.cn/kaijiang/ssq/2025017.html"
    #https://datachart.500.com/ssq/history/history.shtml
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    get(url, headers)
    '''
    get_sales()