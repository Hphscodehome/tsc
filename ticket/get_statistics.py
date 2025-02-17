import numpy as np
import matplotlib.pyplot as plt
import logging,json
import statistics


def open_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    _keys = list(results.keys())# 全部
    _keys = sorted(_keys,reverse=True)# 全部
    _keys = _keys[:-100]# 去掉前一百个的部分
    
    tongjis = []
    qihaos = []
    tongjis_6 = []
    tongjis_5 = []
    for key in _keys:
        # 售出
        shouchu = results[key]['sales']/2
        # 回收
        huishou = 0
        for index_key in results[key].keys():
            if index_key.endswith('等奖'):
                huishou += results[key][index_key]
                
        if huishou != 0:
            tongjis_6.append(results[key]['六等奖']/shouchu)
            tongjis_5.append((results[key]['六等奖']+results[key]['五等奖'])/shouchu)
            tongjis.append(huishou/shouchu)
            qihaos.append(key)
        else:
            logging.info(f"qihao: {key}")
            tongjis_6.append(0)
            tongjis_5.append(0)
            tongjis.append(0)
            qihaos.append(key)
            
    #logging.info(tongjis)
    #logging.info(tongjis_5)
    #logging.info(tongjis_6)
    
    tongjis_6 = np.array(tongjis_6)
    tongjis_5 = np.array(tongjis_5)
    tongjis = np.array(tongjis)
    qihaos = np.array(qihaos,dtype=object)
    
    bigresults = tongjis[tongjis > 1/16]
    smallresults = tongjis[tongjis <= 1/16]
    bigqihaos = qihaos[tongjis > 1/16]
    smallqihaos = qihaos[tongjis <= 1/16]
    
    sorted_tongji = sorted(tongjis)
    sorted_6 = sorted(tongjis_6)
    sorted_5 = sorted(tongjis_5)
    
    logging.info(f"""
统计中位数是：{(sorted_tongji[len(sorted_tongji)//2]+sorted_tongji[len(sorted_tongji)//2-1])/2 if len(sorted_tongji) % 2 == 0 else sorted_tongji[len(sorted_tongji)//2]}
统计第一三分位数：{np.percentile(sorted_tongji, 33.33)}， 统计第二三分位数：{np.percentile(sorted_tongji, 66.67)}
""")
    for i,key in enumerate([1/16,
                (sorted_tongji[len(sorted_tongji)//2]+sorted_tongji[len(sorted_tongji)//2-1])/2 if len(sorted_tongji) % 2 == 0 else sorted_tongji[len(sorted_tongji)//2],
                np.percentile(sorted_tongji, 33.33),
                np.percentile(sorted_tongji, 66.67)]):
        with open(f"./data/artificial_issues_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in qihaos[tongjis<=key]:
                f.write(item + "\n")
    '''
    logging.info(f"""大于1/16的占比有：{len(bigresults)/len(tongjis)}
                 小于1/16的占比有：{len(smallresults)/len(tongjis)}
                 大于1/16的期号是：{bigqihaos}
                 小于1/16的期号是：{smallqihaos}
                 六等奖大于1/16的占比有：{len(tongjis_6[tongjis_6>1/16])/len(tongjis_6)}
                 六等奖小于1/16的占比有：{len(tongjis_6[tongjis_6<=1/16])/len(tongjis_6)}
                 六等奖大于1/16的期号是：{qihaos[tongjis_6>1/16]}
                 六等奖小于1/16的期号是：{qihaos[tongjis_6<=1/16]}
                 五等奖大于1/16的占比有：{len(tongjis_5[tongjis_5>1/16])/len(tongjis_5)}
                 五等奖小于1/16的占比有：{len(tongjis_5[tongjis_5<=1/16])/len(tongjis_5)}
                 五等奖大于1/16的期号是：{qihaos[tongjis_5>1/16]}
                 五等奖小于1/16的期号是：{qihaos[tongjis_5<=1/16]}
                 重合量：{len(list(set(qihaos[tongjis_5<=1/16]) & set(qihaos[tongjis_6<=1/16])))}
                 六级：{len(qihaos[tongjis_6<=1/16])}
                 五级：{len(qihaos[tongjis_5<=1/16])}
                 六级中位数：{(sorted_6[len(sorted_6)//2]+sorted_6[len(sorted_6)//2-1])/2 if len(sorted_6) %2 ==0 else sorted_6[len(sorted_6)//2]}
                 五级中位数：{(sorted_5[len(sorted_5)//2]+sorted_5[len(sorted_5)//2-1])/2 if len(sorted_5) %2 ==0 else sorted_5[len(sorted_5)//2]}
                 六级第一三分位数：{np.percentile(sorted_6, 33.33)}， 六级第二三分位数：{np.percentile(sorted_6, 66.67)}
                 五级第一三分位数：{np.percentile(sorted_5, 33.33)}， 五级第二三分位数：{np.percentile(sorted_5, 66.67)}
                 """)
    
    # 1/16 中位数 第一三分位 第二三分位 
    for i,key in enumerate([1/16,
                (sorted_6[len(sorted_6)//2]+sorted_6[len(sorted_6)//2-1])/2 if len(sorted_6) %2 ==0 else sorted_6[len(sorted_6)//2],
                np.percentile(sorted_6, 33.33),
                np.percentile(sorted_6, 66.67)]):
        with open(f"./data/artificial_issues6_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in qihaos[tongjis_6<=key]:
                f.write(item + "\n")
    for i,key in enumerate([1/16,
                (sorted_5[len(sorted_5)//2]+sorted_5[len(sorted_5)//2-1])/2 if len(sorted_5) %2 ==0 else sorted_5[len(sorted_5)//2],
                np.percentile(sorted_5, 33.33),
                np.percentile(sorted_5, 66.67)]):
        with open(f"./data/artificial_issues5_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in qihaos[tongjis_5<=key]:
                f.write(item + "\n")
    '''
    plot_average(tongjis,figure1='/data/hupenghui/Self/tsc/ticket/data/tongji_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji_figure2.png')
    plot_average(tongjis_6,figure1='/data/hupenghui/Self/tsc/ticket/data/tongji6_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji6_figure2.png')
    plot_average(tongjis_5,figure1='/data/hupenghui/Self/tsc/ticket/data/tongji5_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji5_figure2.png')
    
def plot_average(data, figure1='figure1.png', figure2='figure2.png'):
    """
    绘制两幅曲线图，并添加 y=1/16 的水平线。
    Args:
        data: 输入列表。
    """
    averages1 = []
    for i in range(1, len(data) + 1):
        subset = data[:i]
        average = sum(subset) / len(subset)
        averages1.append(average)
    plt.plot(range(1, len(data) + 1), averages1)
    plt.xlabel("Number of Elements")
    plt.ylabel("Average Value")
    plt.title("Average of First n Elements")
    plt.axhline(y=1/16, color='r', linestyle='--', label='y=1/16')  # 添加水平线
    plt.legend()  # 显示图例
    plt.savefig(figure1)
    plt.close()

    averages2 = []
    for i in range(len(data) - 1, -1, -1):
        subset = data[i:]
        average = sum(subset) / len(subset)
        averages2.append(average)
    plt.plot(range(1, len(data) + 1), averages2)
    plt.xlabel("Number of Elements")
    plt.ylabel("Average Value")
    plt.title("Average of Last n Elements")
    plt.axhline(y=1/16, color='r', linestyle='--', label='y=1/16')  # 添加水平线
    plt.legend()  # 显示图例
    plt.savefig(figure2)
    plt.close()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    file = './data/issue_values.json'
    open_file(file)