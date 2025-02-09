#region other-package
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import kl_divergence, Categorical
from collections import defaultdict
import logging
import asyncio
import pdb
import numpy as np
#endregion

#region my-package
from model.define_model import * #feature_specific_Model_actor,feature_specific_Model_critic
from registry.define_registry import Registry
from utils.constants import obs_fn
#endregion

class World_agent():
    def __init__(self,intersections):
        self.actors = {}
        self.critics = {}
        self.actors_optimizer = {}
        self.critics_optimizer = {}
        for inter in intersections:
            kwargs = {
                'use_func': obs_fn,
                'model_type': 'actor',
                'device': 'cpu',
                'log_dir': './logs'
            }
            self.actors[inter.id] = Registry.mapping['actor']['feature_specific'](**kwargs)
            kwargs = {
                'use_func': obs_fn,
                'model_type': 'critic',
                'device': 'cpu',
                'log_dir': './logs'
            }
            self.critics[inter.id] = Registry.mapping['critic']['feature_specific'](**kwargs)
            self.actors_optimizer[inter.id] = optim.SGD(self.actors[inter.id].parameters(), lr=0.1)
            self.critics_optimizer[inter.id] = optim.SGD(self.critics[inter.id].parameters(), lr=0.0001)
            
    def step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        for inter_id in list(self.actors.keys()):
            actions[inter_id] = self.actors[inter_id].forward(obs[inter_id])
        return actions
    
    async def optimize(self, records):
        tasks = [self.optimize_inter(inter_id, records[inter_id]) for inter_id in self.actors.keys()]
        await asyncio.gather(*tasks)
        #pdb.set_trace()
        return True
            
    async def optimize_inter(self, inter_id, records):
        #pdb.set_trace()
        await self.optimize_critic(inter_id, records)
        await self.optimize_actor(inter_id, records)
        #pdb.set_trace()
        return True
        
    async def optimize_actor(self, inter_id, records):
        """records:dict
        'b_state':[st1,st2,st3]
        'reward':
        'action':
        'a_state':
        'done':
        """
        #计算该状态下对应动作出现的概率。
        #计算该状态下对应动作的优势。
        #求导优化
        #pdb.set_trace()
        actor = self.actors[inter_id]
        critic = self.critics[inter_id]
        actor_optimizer = self.actors_optimizer[inter_id]
        flag = True# 自定义的
        num_epochs = 3
        batch_size = 40
        if flag:
            for epoch in range(num_epochs):
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']),batch_size):
                    indices = permutation[i:i+batch_size]
                    temp_records = np.array(records['b_state'], dtype=object)[indices]
                    #pdb.set_trace()
                    actions = actor.forward_batch(temp_records)
                    #pdb.set_trace()
                    with torch.no_grad():
                        temp = 0
                        for _ in range(5):
                            new_records = []
                            for recor in temp_records:
                                new_recor = {}
                                for key in recor.keys():
                                    if key != 'mask' and 'phase' not in key:
                                        new_recor[key] = recor[key] + np.random.randn(*recor[key].shape)
                                    else:
                                        new_recor[key] = recor[key]
                                new_records.append(new_recor)
                            #pdb.set_trace()
                            noisy_actions = actor.forward_batch(new_records)
                            temp += noisy_actions
                        expected_action = temp/5
                    #pdb.set_trace()
                    batch_size, num_samples, num_features = actions.shape  # 这里 num_features 应该是 2
                    actions_l1 = F.softmax(actions[..., 0], dim=1)  # dim=1 因为我们要在第二个维度上应用 softmax
                    expected_l1 = F.softmax(expected_action[..., 0], dim=1)
                    dist1 = Categorical(probs=actions_l1)
                    dist2 = Categorical(probs=expected_l1)
                    kl_div = kl_divergence(dist1, dist2)  # 对样本维度求和
                    
                    #pdb.set_trace()
                    
                    actions_l2 = F.sigmoid(actions[..., 1])
                    expected_l2 = F.sigmoid(expected_action[..., 1])
                    # 计算距离
                    distance = (actions_l2 * torch.log(actions_l2 / expected_l2) + \
                                (1 - actions_l2) * torch.log((1 - actions_l2) / (1 - expected_l2))) * expected_l1
                    # 现在我们需要对样本和批量进行求和
                    distance = distance.sum(dim=1) + kl_div
                    # 如果你需要一个标量值，可以继续对批量进行求和
                    distance = distance.sum()
                    
                    action_prob = F.sigmoid(distance)
                    with torch.no_grad():
                        value = critic.forward_batch(np.array(records['b_state'], dtype=object)[indices])
                    loss = sum(torch.log(action_prob)*value.flatten())
                    logging.info(f"{loss},{[key for key in actor.output_layer.parameters()]}")
                    actor_optimizer.zero_grad()
                    loss.backward()
                    actor_optimizer.step()
                    logging.info(f"{loss},{[key for key in actor.output_layer.parameters()]}")
        #pdb.set_trace()
        return True
                    
    async def optimize_critic(self, inter_id, records):
        """records:dict
        'b_state':[st1,st2,st3]
        'reward':
        'action':
        'a_state':
        'done':
        """
        #计算该状态下对应状态的状态价值
        #计算下个状态的状态价值
        #根据reward调整状态价值函数
        #pdb.set_trace()
        critic = self.critics[inter_id]
        critic_optimizer = self.critics_optimizer[inter_id]
        flag = True # 自定义的
        num_epochs = 3
        batch_size = 40
        if flag:
            for epoch in range(num_epochs):
                #pdb.set_trace()
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']), batch_size):
                    indices = permutation[i:i+batch_size]
                    temp_records = np.array(records['b_state'], dtype=object)[indices]
                    #pdb.set_trace()
                    values = critic.forward_batch(temp_records)
                    
                    with torch.no_grad():
                        temp_records = np.array(records['a_state'], dtype=object)[indices]
                        expected_values = critic.forward_batch(temp_records)
                    #pdb.set_trace()
                    expected_values = expected_values + torch.from_numpy(np.array(records['reward'])[indices]).float().reshape(expected_values.shape)
                    loss = sum((expected_values-values)**2)/batch_size
                    critic_optimizer.zero_grad()
                    loss.backward()
                    critic_optimizer.step()
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{Registry.mapping}")
    logging.info(f"nihao")
    True