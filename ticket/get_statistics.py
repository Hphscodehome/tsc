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
    for key in _keys:
        # 售出
        shouchu = results[key]['sales']/2
        # 回收
        huishou = 0
        for index_key in results[key].keys():
            if index_key.endswith('等奖'):
                huishou += results[key][index_key]
                
        if huishou != 0:
            tongjis.append(huishou/shouchu)
            qihaos.append(key)
        else:
            logging.info(f"qihao: {key}")
            tongjis.append(0)
            qihaos.append(key)
            
    #logging.info(tongjis)
    
    tongjis = np.array(tongjis)
    qihaos = np.array(qihaos,dtype=object)
    
    others = 0.01
    for _ in range(len(tongjis)):
        tongjis[_] = max(tongjis[_] - others,0) # 真实的统计值
    
    bigresults = tongjis[tongjis > 1/16]
    smallresults = tongjis[tongjis <= 1/16]

    sorted_tongji = sorted(tongjis)
    
    logging.info(f"""
总的来说，大于1/16的占比有：{len(bigresults)/len(tongjis)}
总的来说，小于1/16的占比有：{len(smallresults)/len(tongjis)}
统计结果如下所示：
统计中位数是：{np.percentile(sorted_tongji, 50)}
统计第一三分位数：{np.percentile(sorted_tongji, 33.33)}， 统计第二三分位数：{np.percentile(sorted_tongji, 66.67)}
统计第一四分位数：{np.percentile(sorted_tongji, 25)}， 统计第3四分位数：{np.percentile(sorted_tongji, 75)}
统计40分位数：{np.percentile(sorted_tongji, 40)}
""")
    
    plot_average(tongjis, split = 1/16, figure1='/data/hupenghui/Self/tsc/ticket/data/tongji_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji_figure2.png')
    
    for i,key in enumerate([1/16,
                np.percentile(sorted_tongji, 50),
                np.percentile(sorted_tongji, 40),
                np.percentile(sorted_tongji, 60)]):
        logging.info(f"{key},./data/artificial_issues_{str(i)}.txt")
        with open(f"./data/artificial_issues_{str(i)}.txt", 'w', encoding='utf-8') as f:
            for item in qihaos[tongjis<=key]:
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