import torch
from torch.distributions import Categorical
import logging
import argparse
from model_data_v2 import Model
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="命令行参数截断值")
    parser.add_argument("-e", "--end", type=int, help="截断值",default=9)
    args = parser.parse_args()
    logging.info(f"当前试验截断值为：{args.end}")
    model = Model()
    
    #test = torch.tensor([11,1,13,14,11,5,9,10,2,9,9,10,15,1,3,15,10,14,3,8,5,10,10,5,7,10,10,13,4,14]).to(torch.int64)
    #end = torch.randint(args.end-1, len(test), (1,)).item()
    #temp = test[end-(args.end-1):end+1].unsqueeze(0)
    #test=temp
    test = torch.tensor([11,1,13,14,11,5,9,10,2,9,9,10,15,1,3,15,10,14,3,8,5,10,10,5,7,10,10,13,4,14]).to(torch.int64)
    test = test[-args.end:].unsqueeze(0)
    checkpoint_path = f'/data/hupenghui/Self/tsc/ticket/model22/best_model_{args.end}.pth'
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict)
    #for name, param in model.named_parameters():
    #    print(f"参数名称: {name}, 参数形状: {param.shape}, 参数值：{param}")
    with torch.no_grad():
        outputs = model(test)
        #outputs = model(torch.from_numpy(np.array(list(dataset.x[0][1:])+[int(dataset.y[0])], dtype=np.int64)).to(torch.int).unsqueeze(0))
    distribution = Categorical(logits=outputs)
    samples = distribution.sample()
    logging.info(f"INput:{test.flatten().tolist()},\n当前模型预测结果为：{outputs.flatten().tolist()},\nlogits最大的值为：{torch.argmax(outputs,dim=1).item()+1}\n按照logits采样得到的结果是：{samples.item()+1}")