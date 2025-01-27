import torch
from utils.str_int import get_int,get_char
from utils.constants import Chars
phase_str = 'rgrg'
action = torch.randn(len(phase_str),2)
lanes_conflict_map = torch.randint(0,2,size=(4,4))
def get_phase(action):
    # action lanes*2
    # lanes*1是link重要性
    # lanes*2是link变不变
    # 结合冲突车道确定下一个可以选择的link。
    result = ['' for _ in range(len(phase_str))]
    mask = torch.tensor([False for _ in range(len(phase_str))])
    logits = torch.tensor(action[:,0])
    while '' in result:
        filtered_logits = logits[~mask]  # 取反mask，保留False对应的logits
        indices = torch.arange(len(mask))[~mask]  # 获取mask为False的索引
        distribution = torch.distributions.Categorical(logits=filtered_logits)
        lane_sample = distribution.sample().item()
        lane_sample = indices[lane_sample].item()
        id = get_int(phase_str[lane_sample])
        probability = torch.sigmoid(action[lane_sample,1])
        distribution = torch.distributions.Bernoulli(probability)# 创建一个Bernoulli分布
        change_sample = distribution.sample().item()
        if change_sample == 1:
            id += 1
            id = id % Chars
        lane_char = get_char(id)
        result[lane_sample] = lane_char
        if lane_char != 'r':
            conflict_lanes =(torch.tensor(lanes_conflict_map[lane_sample,:])>0)
            mask = mask | conflict_lanes
            for index,flag in enumerate(conflict_lanes):
                if flag:
                    result[index] = 'r'
        mask[lane_sample] = True
    print(result)
get_phase(action)