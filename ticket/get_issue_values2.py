import requests
from bs4 import BeautifulSoup
import logging,json,re

index_values = {}
def from_datachart500():
    global index_values
    url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=00000&end=99999'
    response = requests.get(url)
    response.encoding = 'utf-8'
    response = response.text
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table', id='tablelist')
    for row in table.find_all('tr', class_='t_tr1'):
        cells = row.find_all('td')
        issue = int('20' + cells[0].text.strip()) #期号
        value = int(cells[14].text.replace(",", "").strip()) # 投注金额
        left = int(cells[9].text.replace(",", "").strip()) # 资金池剩余资金
        yidengrenshu = int(cells[10].text.replace(",", "").strip()) # 一等人数
        yidengjiang = int(cells[11].text.replace(",", "").strip()) # 一等奖金
        erdengrenshu = int(cells[12].text.replace(",", "").strip()) # 二等人数
        erdengjiang = int(cells[13].text.replace(",", "").strip()) # 二等奖金
        reb_balls = [int(cells[i].text.strip()) for i in range(1,7)] # 红球
        blue_balls = [int(cells[7].text.strip())] # 蓝球
        if len(reb_balls) != 6 or len(blue_balls) != 1:
            logging.info(f"Error: {issue}")
        result = from_zx500(url = f"https://zx.500.com/ssq/{str(issue)}/")
        if result != None:
            if (set(reb_balls) == set(result['red'])) and \
                (set(blue_balls) == set(result['blue'])) and \
                (left == result['left']) and \
                (yidengrenshu == result['prize']['一等奖']['winners']) and \
                (yidengjiang == result['prize']['一等奖']['prize_amount']) and \
                (erdengrenshu == result['prize']['二等奖']['winners']) and \
                (erdengjiang == result['prize']['二等奖']['prize_amount']):
                index_values[issue] = {}
                index_values[issue]['red'] = reb_balls
                index_values[issue]['blue'] = blue_balls[0]
                index_values[issue]['sales'] = value
                for key in result['prize'].keys():
                    index_values[issue][key] = result['prize'][key]['winners']
        else:
            result = from_vipc(url = f"https://www.vipc.cn/result/ssq/{str(issue)}")
            if (set(reb_balls) == set(result['red'])) and \
                (set(blue_balls) == set(result['blue'])) and \
                (left == result['left']) and \
                (yidengrenshu == result['prize']['一等奖']['winners']) and \
                (yidengjiang == result['prize']['一等奖']['prize_amount']) and \
                (erdengrenshu == result['prize']['二等奖']['winners']) and \
                (erdengjiang == result['prize']['二等奖']['prize_amount']) and \
                (value == result['sales']):
                index_values[issue] = {}
                index_values[issue]['red'] = reb_balls
                index_values[issue]['blue'] = blue_balls[0]
                index_values[issue]['sales'] = value
                for key in result['prize'].keys():
                    index_values[issue][key] = result['prize'][key]['winners']

# "https://zx.500.com/ssq/2025016/"       
def from_zx500(url = "https://zx.500.com/ssq/2025016/"):
    logging.info(f"url: {url}")
    try:
        result = {}
        response = requests.get(url)
        response.encoding = 'gb2312'
        html_content = response.text
        #print(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        draw_numbers = soup.find('ul', class_='wnum').find_all('li')
        red_balls = [int(ball.text.strip()) for ball in draw_numbers if 'redball' in ball['class']]
        blue_ball = [int(ball.text.strip()) for ball in draw_numbers if 'blueball' in ball['class']]#draw_numbers[-1].text
        if len(red_balls) != 6 or len(blue_ball) != 1:
            logging.info(f"Error: {url}")
        result['red'] = red_balls
        result['blue'] = blue_ball
        pool_info = int(soup.find('div', class_='gc').find('span').text.replace('元', '').replace(',', '').strip())
        result['left'] = pool_info
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
def from_vipc(url = 'https://www.vipc.cn/result/ssq/2014001' , headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}):
    logging.info(f"url: {url}")
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
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from_datachart500()
    with open('/data/hupenghui/Self/tsc/ticket/data/issue_values.json', 'w', encoding='utf-8') as f:
        json.dump(index_values, f, indent=4)  # indent=4 表示缩进 4 个空格
    #print(from_vipc())
    #print(from_zx500())