import numpy as np
import matplotlib.pyplot as plt
import logging,json
import statistics

all_keys = ['一等奖','二等奖','三等奖','四等奖','五等奖','六等奖']
noise = 0.01
train_start = 100

def get_shouchu_huishou(_keys,results):
    tongjis = []
    qihaos = []
    for key in _keys:
        shouchu = results[key]['sales']/2
        huishou = 0
        for index_key in all_keys:
            huishou += results[key][index_key]
        if huishou != 0:
            tongjis.append(huishou/shouchu)
            qihaos.append(key)
        else:
            logging.info(f"qihao: {key}")
            tongjis.append(0)
            qihaos.append(key)
    return tongjis,qihaos
    
def open_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    _keys = list(results.keys())# 全部
    _keys = sorted(_keys,reverse=True)# 全部 从大到小
    
    Total_keys = _keys
    Total_tongjis,Total_qihaos = get_shouchu_huishou(Total_keys,results)
    Total_tongjis = np.array(Total_tongjis)
    Total_qihaos = np.array(Total_qihaos,dtype=object)
    for _ in range(len(Total_tongjis)):
        Total_tongjis[_] = max(Total_tongjis[_] - noise,0) # 真实的统计值
        
    Total_bigresults = Total_tongjis[Total_tongjis > 1/16]
    Total_smallresults = Total_tongjis[Total_tongjis <= 1/16]
    
    
    Partial_keys = _keys[:-train_start]# 去掉前一百个的部分
    Partial_tongjis,Partial_qihaos = get_shouchu_huishou(Partial_keys,results)
    Partial_tongjis = np.array(Partial_tongjis)
    Partial_qihaos = np.array(Partial_qihaos,dtype=object)
    for _ in range(len(Partial_tongjis)):
        Partial_tongjis[_] = max(Partial_tongjis[_] - noise,0) # 真实的统计值
    
    Partial_bigresults = Partial_tongjis[Partial_tongjis > 1/16]
    Partial_smallresults = Partial_tongjis[Partial_tongjis <= 1/16]
    
    sorted_Partial_tongjis = sorted(Partial_tongjis)
    
    
    logging.info(f"""
扣除前百数据后的统计结果：
大于1/16的占比有：{len(Partial_bigresults)/len(Partial_tongjis)}
小于1/16的占比有：{len(Partial_smallresults)/len(Partial_tongjis)}
全部数据的统计结果：
大于1/16的占比有：{len(Total_bigresults)/len(Total_tongjis)}
小于1/16的占比有：{len(Total_smallresults)/len(Total_tongjis)}
统计结果如下所示：
统计中位数是：{np.percentile(sorted_Partial_tongjis, 50)}
统计第一三分位数：{np.percentile(sorted_Partial_tongjis, 33.33)}
统计第二三分位数：{np.percentile(sorted_Partial_tongjis, 66.67)}
统计第一四分位数：{np.percentile(sorted_Partial_tongjis, 25)}
统计第3四分位数：{np.percentile(sorted_Partial_tongjis, 75)}
统计40分位数：{np.percentile(sorted_Partial_tongjis, 40)}
""")
    logging.info(f"{Partial_tongjis[:10]}")
    
    for i,key in enumerate([1/16,
                np.percentile(sorted_Partial_tongjis, 50),
                np.percentile(sorted_Partial_tongjis, 40),
                np.percentile(sorted_Partial_tongjis, 60)]):
        logging.info(f"{key},./data/artificial_issues_{str(i)}.txt")
        with open(f"./data/artificial_issues_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in Partial_qihaos[Partial_tongjis<=key]:
                f.write(item + "\n")
        
        logging.info(f"{key},./data/artificial_issuestot_{str(i)}.txt")
        with open(f"./data/artificial_issuestot_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in Total_qihaos[Total_tongjis<=key]:
                f.write(item + "\n")

def plot_average(data, split, figure1='figure1.png', figure2='figure2.png'):
    """
    绘制两幅曲线图，并添加 y=1/16 的水平线。
    Args:
        data: 输入列表。
    """
    averages1 = []
    for i in range(1, len(data) + 1):
        subset = data[:i]
        average = 1-2*len(subset[subset>split])/len(subset)
        averages1.append(average)
    plt.semilogx(range(1, len(data) + 1), averages1)
    plt.xlabel("Number of Elements")
    plt.ylabel("Average Value")
    plt.title("Average of First n Elements")
    plt.axhline(y=1/16, color='r', linestyle='--', label='y=1/16')  # 添加水平线
    plt.legend()  # 显示图例
    for i in range(min(10, len(data))):
        plt.text(i+1, averages1[i], f'({i+1}, {averages1[i]:.2f})', 
                fontsize=9, ha='right', va='bottom')
    plt.savefig(figure1)
    plt.close()

    averages2 = []
    for i in range(len(data) - 1, -1, -1):
        subset = data[i:]
        average = 1-2*len(subset[subset>split])/len(subset)
        averages2.append(average)
    plt.semilogx(range(1, len(data) + 1), averages2)
    plt.xlabel("Number of Elements")
    plt.ylabel("Average Value")
    plt.title("Average of Last n Elements")
    plt.axhline(y=1/16, color='r', linestyle='--', label='y=1/16')  # 添加水平线
    plt.legend()  # 显示图例
    for i in range(min(10, len(data))):
        plt.text(i+1, averages2[i], f'({i+1}, {averages2[i]:.2f})', 
                fontsize=9, ha='right', va='bottom')
    plt.savefig(figure2)
    plt.close()
              
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    file = './data/issue_values.json'
    open_file(file)