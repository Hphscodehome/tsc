import torch
from torch.distributions import Categorical
import logging,json
import argparse
from model_data_v2 import Model
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    model = Model()
    file = '/data/hupenghui/Self/tsc/ticket/data/issue_values.json'
    with open(file, 'r', encoding='utf-8') as f:
        results = json.load(f)#无序
    all_indexs = list(results.keys())
    all_indexs = sorted(all_indexs)# 从小到大
    numbers = [int(results[key]['blue']) for key in all_indexs[-100:]]
    logging.info(f"原始测试数据(15个)是：{numbers[-15:]}")
    test = torch.tensor(numbers).to(torch.int64)
    test -= 1
    test = test[-args.end:].unsqueeze(0)
    logging.info(f"真实测试数据是：{test.flatten()}")
    
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model2/best_model_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict)
    #for name, param in model.named_parameters():
    #    print(f"参数名称: {name}, 参数形状: {param.shape}, 参数值：{param}")
    with torch.no_grad():
        outputs = model(test)
    distribution = Categorical(logits=outputs)
    samples = distribution.sample((10,))
    logging.info(f"""
输入数据是:{test.flatten().tolist()},
当前模型预测结果为：
{outputs.flatten().tolist()},
logits转化为概率分布是:
{distribution.probs.flatten().tolist()},
logits最大的值为:
{torch.argmax(outputs,dim=1).item()+1}
按照logits采样得到的结果是:
{samples.flatten()+1}
""")