import numpy as np
import matplotlib.pyplot as plt
import logging,json

def open_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # 计算蓝色球基本概率
    _keys = list(results.keys())
    _keys = sorted(_keys,reverse=True)
    tongjis = []
    qihaos = []
    tongjis_6 = []
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
            tongjis.append(huishou/shouchu)
            qihaos.append(key)
        else:
            tongjis_6.append(0)
            tongjis.append(0)
            qihaos.append(key)
    logging.info(tongjis)
    logging.info(tongjis_6)
    tongjis_6 = np.array(tongjis_6)
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
                 六等奖大于1/16的占比有：{len(tongjis_6[tongjis_6>1/16])/len(tongjis_6)}
                 六等奖小于1/16的占比有：{len(tongjis_6[tongjis_6<=1/16])/len(tongjis_6)}
                 六等奖大于1/16的期号是：{qihaos[tongjis_6>1/16]}
                 六等奖小于1/16的期号是：{qihaos[tongjis_6<=1/16]}
                 """)
    with open('/data/hupenghui/Self/tsc/ticket/data/artificial_issues.txt', 'w', encoding='utf-8') as f:
        for item in qihaos[tongjis_6<=1/16]:
            f.write(item + "\n")
            
    plot_average(tongjis,figure1='/data/hupenghui/Self/tsc/ticket/data/tongji_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji_figure2.png')
    plot_average(tongjis_6,figure1='/data/hupenghui/Self/tsc/ticket/data/tongji6_figure1.png', figure2='/data/hupenghui/Self/tsc/ticket/data/tongji6_figure2.png')
    
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
    file = '/data/hupenghui/Self/tsc/ticket/data/issue_values.json'
    open_file(file)