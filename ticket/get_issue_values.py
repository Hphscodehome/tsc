from bs4 import BeautifulSoup
import logging,json,re,requests
import numpy as np

#Dict[index,Dict[attr,value]]
#attr in [red,blue,6,5,4,3,2,1,sales]
index_values = {}

# https://m.78500.cn/kaijiang/ssq/2025017.html
# 09年以后的才有
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
            winners = soup.select('.winner-list tbody tr')
            for winner in winners:
                cells = winner.find_all('td')
                if len(cells) == 3:
                    award_name = cells[0].text.strip()
                    count = cells[1].text.strip()
                    index_values[number][award_name[:3]] = int(count)
                    #logging.info(f"{award_name}: {count} 注，每注 {amount} 元")
            logging.info(f"{number},{index_values[number]}")
            return True
        else:
            return False
    else:
        #logging.info(f"请求失败，状态码：{response.status_code}")
        return False


# https://www.vipc.cn/result/ssq/2025016
# 2013年之后的才有
def from_vipc(url = 'https://www.vipc.cn/result/ssq/2014001' , headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}):
    #logging.info(f"url: {url}")
    result = {}
    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    red_balls = [int(ball.text.replace(",", "").strip()) for ball in soup.select('.vRes_lottery_ball b.red')]
    blue_balls = [int(ball.text.replace(",", "").strip()) for ball in soup.select('.vRes_lottery_ball b.blue')]
    if len(red_balls) != 6 or len(blue_balls) != 1:
        logging.info(f"Error: {url}")
    result['red'] = red_balls
    result['blue'] = blue_balls
    divs = soup.find_all('div', class_='vResult_contentDigit_main_item')
    for div in divs:
        if div:
            text = div.text
            name = text.split('：')[0]
            amount_part = text.split('：')[1]
            if '奖池滚存' in name:
                result['left'] = int(amount_part.strip().replace('元', '').replace(',', '').strip())
            elif '本期销量' in name:
                result['sales'] = int(amount_part.strip().replace('元', '').replace(',', '').strip())
    prizes = {}
    table_rows = soup.select('.vResult_contentDigit_table tr')
    for row in table_rows[1:]:  # Skip the header row
        cells = row.find_all('td')
        if len(cells) == 3:
            prizes[cells[0].text.strip()] = {}
            prizes[cells[0].text.strip()]['winners'] = int(cells[1].text.replace(",", "").strip())
            prizes[cells[0].text.strip()]['prize_amount'] = int(cells[2].text.replace(",", "").strip())
    result['prize'] = prizes
    return result

def get_sales():
    
    global index_values
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=09111&end=99999'#"http://datachart.500.com/ssq/history/newinc/history.php?start=00001&end=2024097"#'https://datachart.500.com/ssq/history/outball.shtml'
    response = requests.get(url)
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    for row in table.find_all('tr', class_='t_tr1'):  # 使用 class_ 过滤
        cells = row.find_all('td')
        issue = int('20' + cells[0].text.strip())
        value = int(cells[14].text.replace(",", "").strip())
        index_values[issue] = {}
        index_values[issue]['sales'] = value
        index_values[issue]['red'] = [cells[i].text.strip() for i in range(1,7)]
        index_values[issue]['blue'] = cells[7].text.strip()
        
def process(results):
    _keys = list(results.keys())
    _keys = sorted(_keys,reverse=True)
    tongjis = []
    qihaos = []
    for key in _keys:
        # 售出
        shouchu = results[key]['sales']/2
        # 回收
        huishou = 0
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
    get_sales()
    logging.info(f"{index_values}")
    for key in index_values.keys():
        url = f"https://m.78500.cn/kaijiang/ssq/{str(key)}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        get(url, headers)
    logging.info(f"{index_values}")
    with open('/data/hupenghui/Self/tsc/ticket/data/issue_values.json', 'w', encoding='utf-8') as f:
        json.dump(index_values, f, indent=4)  # indent=4 表示缩进 4 个空格
        
    process(index_values)
    # 2009111
    '''
    url = "https://m.78500.cn/kaijiang/ssq/2025017.html"
    #https://datachart.500.com/ssq/history/history.shtml
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    get(url, headers)
    '''
    