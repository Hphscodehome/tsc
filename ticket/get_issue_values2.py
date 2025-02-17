import requests
from bs4 import BeautifulSoup
import logging,json,re,time
from ratelimit import limits, sleep_and_retry

index_values = {}
all_keys = ['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖']

def from_datachart500():
    global index_values
    with open('./data/issue_values.json', 'r', encoding='utf-8') as f:
        index_values = json.load(f)# 全部
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=00000&end=99999'
    response = requests.get(url,
                            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
    response.encoding = 'utf-8'
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    for row in table.find_all('tr', class_='t_tr1'):
        cells = row.find_all('td')
        issue = '20' + cells[0].text.strip() #期号
        value = int(cells[14].text.replace(",", "").strip()) # 投注金额
        left = int(cells[9].text.replace(",", "").strip()) # 资金池剩余资金
        yidengrenshu = int(cells[10].text.replace(",", "").strip()) # 一等人数
        yidengjiang = int(cells[11].text.replace(",", "").strip()) # 一等奖金
        erdengrenshu = int(cells[12].text.replace(",", "").strip()) # 二等人数
        erdengjiang = int(cells[13].text.replace(",", "").strip()) # 二等奖金
        #print(yidengrenshu,yidengjiang,erdengrenshu,erdengjiang)
        reb_balls = [int(cells[i].text.strip()) for i in range(1,7)] # 红球
        blue_balls = [int(cells[7].text.strip())] # 蓝球
        if len(reb_balls) != 6 or len(blue_balls) != 1:
            logging.info(f"Error: {issue}")
            break
        if str(issue) in index_values.keys():
            continue
        if fetch_lottery_results(issue, reb_balls, blue_balls, value, all_keys, yidengrenshu, yidengjiang, erdengrenshu, erdengjiang):
            continue
        else:
            break
        
        
def fetch_lottery_results(issue, reb_balls, blue_balls, value, all_keys, yidengrenshu, yidengjiang, erdengrenshu, erdengjiang):
    global index_values
    if issue > '2013001':
        urls = [
            f"https://m.78500.cn/kaijiang/ssq/{issue}.html",
            f"https://www.vipc.cn/result/ssq/{issue}",
            f"https://zx.500.com/ssq/{issue}/"
        ]
        fetchers = [from_78500, from_vipc, from_zx500]
        for url, fetch in zip(urls, fetchers):
            result = fetch(url=url)
            if result is not None and set(all_keys) <= set(result['prize'].keys()):
                if any([
                    yidengrenshu == result['prize']['一等奖']['winners'],
                    yidengjiang == result['prize']['一等奖']['prize_amount'],
                    erdengrenshu == result['prize']['二等奖']['winners'],
                    erdengjiang == result['prize']['二等奖']['prize_amount']
                ]):
                    index_values[issue] = {
                        'red': reb_balls,
                        'blue': blue_balls[0],
                        'sales': value,
                        **{key: result['prize'][key]['winners'] for key in all_keys}
                    }
                    return True# Found the data, exit function
                
            logging.info(f"{url}, not found, {issue}")
        # If we've gone through all sources without finding valid data
        logging.info(f"没有记录: {issue}")
        return False
    
    elif issue > '2009111':
        urls = [
            f"https://m.78500.cn/kaijiang/ssq/{issue}.html",
            f"https://zx.500.com/ssq/{issue}/"
        ]
        fetchers = [from_78500, from_zx500]
        for url, fetch in zip(urls, fetchers):
            result = fetch(url=url)
            if result is not None and set(all_keys) <= set(result['prize'].keys()):
                if any([
                    yidengrenshu == result['prize']['一等奖']['winners'],
                    yidengjiang == result['prize']['一等奖']['prize_amount'],
                    erdengrenshu == result['prize']['二等奖']['winners'],
                    erdengjiang == result['prize']['二等奖']['prize_amount']
                ]):
                    index_values[issue] = {
                        'red': reb_balls,
                        'blue': blue_balls[0],
                        'sales': value,
                        **{key: result['prize'][key]['winners'] for key in all_keys}
                    }
                    return True # Found the data, exit function

            logging.info(f"{url}, not found, {issue}")
        # If we've gone through all sources without finding valid data
        logging.info(f"没有记录: {issue}")
        return False
    
    else:
        urls = [
            f"https://zx.500.com/ssq/{issue}/"
        ]
        fetchers = [from_zx500]
        for url, fetch in zip(urls, fetchers):
            result = fetch(url=url)
            if result is not None and set(all_keys) <= set(result['prize'].keys()):
                if any([
                    yidengrenshu == result['prize']['一等奖']['winners'],
                    yidengjiang == result['prize']['一等奖']['prize_amount'],
                    erdengrenshu == result['prize']['二等奖']['winners'],
                    erdengjiang == result['prize']['二等奖']['prize_amount']
                ]):
                    index_values[issue] = {
                        'red': reb_balls,
                        'blue': blue_balls[0],
                        'sales': value,
                        **{key: result['prize'][key]['winners'] for key in all_keys}
                    }
                    return True # Found the data, exit function
                
            logging.info(f"{url}, not found, {issue}")
        # If we've gone through all sources without finding valid data
        logging.info(f"没有记录: {issue}")
        return False
        

# https://m.78500.cn/kaijiang/ssq/2025017.html
@sleep_and_retry
@limits(calls=8, period=1)  # 8 call per 1 second
def from_78500(url = "https://m.78500.cn/kaijiang/ssq/2025015.html",
               headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}):
    logging.info(f"url: {url}")
    try:
        result = {}
        response = requests.get(url, headers = headers)
        #print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        numbers = soup.select('.winning-results .number i')
        sales = soup.select_one('.article .gray2:nth-of-type(1)')
        sales_number = sales.text.replace(',', '').strip()
        sales_number = re.search(r'\d+', sales_number).group()
        if numbers and (int(sales_number) != 0):
            #red_balls = [int(ball.text) for ball in soup.select('.c-red')]
            #blue_balls = [int(ball.text) for ball in soup.select('.c-blue')]
            #if len(red_balls) != 6 or len(blue_balls) != 1:
            #    logging.info(f"Error: {url}")
            #result['red'] = red_balls
            #result['blue'] = blue_balls
            # 解析额外信息
            #result['sales'] = int(sales_number)
            #result['left'] = int(soup.select_one('.article p.yellow').text.replace('元', '').replace(',', '').strip().split('：')[1])
            winners = soup.select('.winner-list tbody tr')
            prizes = {}
            for winner in winners:
                cells = winner.find_all('td')
                if len(cells) == 3:
                    award_name = cells[0].text.strip()
                    prizes[award_name[:3]] = {}
                    prizes[award_name[:3]]['winners'] = int(cells[1].text.strip())
                    prizes[award_name[:3]]['prize_amount'] = int(cells[2].text.strip())
            result['prize'] = prizes
            return result
        else:
            return None
    except:
        return None

# "https://zx.500.com/ssq/2025016/"
@sleep_and_retry
@limits(calls=4, period=1)  # 4 call per 1 second
def from_zx500(url = "https://zx.500.com/ssq/2015053/",
               headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}):
    logging.info(f"url: {url}")
    try:
        result = {}
        response = requests.get(url,headers = headers)
        response.encoding = 'gb2312'
        html_content = response.text
        #print(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        #draw_numbers = soup.find('ul', class_='wnum').find_all('li')
        #red_balls = [int(ball.text.strip()) for ball in draw_numbers if 'redball' in ball['class']]
        #blue_ball = [int(ball.text.strip()) for ball in draw_numbers if 'blueball' in ball['class']]#draw_numbers[-1].text
        #if len(red_balls) != 6 or len(blue_ball) != 1:
        #    logging.info(f"Error: {url}")
        #result['red'] = red_balls
        #result['blue'] = blue_ball
        #pool_info = int(soup.find('div', class_='gc').find('span').text.replace('元', '').replace(',', '').strip())
        #result['left'] = pool_info
        prize_table = soup.find('table', class_='seo_tbale')
        prizes = {}
        for row in prize_table.find_all('tr')[2:]:
            columns = row.find_all('td')
            if len(columns) == 3:
                prizes[columns[0].text.replace(',', '').strip()] = {}
                prizes[columns[0].text.replace(',', '').strip()]['winners'] = int(columns[1].text.replace(',', '').strip())
                prizes[columns[0].text.replace(',', '').strip()]['prize_amount'] = int(columns[2].text.replace(',', '').strip())
        result['prize'] = prizes
        return result
    except:
        return None

# https://www.vipc.cn/result/ssq/2025016
# 2013年之后的才有
@sleep_and_retry
@limits(calls=5, period=1)  # 5 call per 1 second
def from_vipc(url = 'https://www.vipc.cn/result/ssq/2003001' , 
              headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}):
    logging.info(f"url: {url}")
    try:
        result = {}
        response = requests.get(url, headers = headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        #red_balls = [int(ball.text.replace(",", "").strip()) for ball in soup.select('.vRes_lottery_ball b.red')]
        #blue_balls = [int(ball.text.replace(",", "").strip()) for ball in soup.select('.vRes_lottery_ball b.blue')]
        #if len(red_balls) != 6 or len(blue_balls) != 1:
        #    logging.info(f"Error: {url}")
        #result['red'] = red_balls
        #result['blue'] = blue_balls
        #divs = soup.find_all('div', class_='vResult_contentDigit_main_item')
        #for div in divs:
        #    if div:
        #        text = div.text
        #        name = text.split('：')[0]
        #        amount_part = text.split('：')[1]
        #        if '奖池滚存' in name:
        #            result['left'] = int(amount_part.strip().replace('元', '').replace(',', '').strip())
        #        elif '本期销量' in name:
        #            result['sales'] = int(amount_part.strip().replace('元', '').replace(',', '').strip())
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
    except:
        return None
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from_datachart500()
    with open('./data/issue_values.json', 'w', encoding='utf-8') as f:
        json.dump(index_values, f, indent=4)  # indent=4 表示缩进 4 个空格
    '''
    issue = '2009001'
    urls = [
        f"https://m.78500.cn/kaijiang/ssq/{issue}.html",
        f"https://www.vipc.cn/result/ssq/{issue}",
        f"https://zx.500.com/ssq/{issue}/"
    ]
    fetchers = [from_78500, from_vipc, from_zx500]
    for url, fetch in zip(urls, fetchers):
        print(fetch(url))
    
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=00000&end=09110'
    response = requests.get(url)
    response.encoding = 'utf-8'
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    for row in table.find_all('tr', class_='t_tr1'):
        cells = row.find_all('td')
        issue = '20' + cells[0].text.strip() #期号
        logging.info(f"{time.time()}")
        logging.info(from_zx500(f"https://zx.500.com/ssq/{issue}/"))
        logging.info(f"{time.time()}")
    '''
    