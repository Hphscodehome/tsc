#region other-package
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import kl_divergence, Categorical
from collections import defaultdict
import logging
import asyncio
#endregion

#region my-package
from model.define_model import feature_specific_Model
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
            self.critics[inter.id] = Registry.mapping['critic']['feature_specific'](**kwargs)
            self.actors_optimizer[inter.id] = optim.SGD(self.actors[inter.id].parameters(),lr=0.001)
            self.critics_optimizer[inter.id] = optim.SGD(self.critics[inter.id].parameters(),lr=0.001)
            
    def step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        for inter_id in list(self.actors.keys()):
            actions[inter_id] = self.actors[inter_id](obs[inter_id])
        return actions
    
    async def optimize(self,records):
        tasks = [self.optimize_inter(inter_id, records[inter_id]) for inter_id in self.actors.keys()]
        results = await asyncio.gather(*tasks)
            
    async def optimize_inter(self, inter_id, records):
        await self.optimize_critic(inter_id, records)
        await self.optimize_actor(inter_id, records)
        
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
        actor = self.actors[inter_id]
        critic = self.critics[inter_id]
        actor_optimizer = self.actors_optimizer[inter_id]
        flag = True# 自定义的
        num_epochs = 100
        batch_size = 40
        if flag:
            for epoch in range(num_epochs):
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']),batch_size):
                    indices = permutation[i:i+batch_size]
                    temp_records = records['b_state'][indices]
                    
                    actions = actor.forward_batch(temp_records)
                    with torch.no_grad():
                        temp = 0
                        for _ in range(5):
                            new_records = []
                            for recor in temp_records:
                                new_recor = {}
                                for key in recor.keys():
                                    if key != 'mask':
                                        new_recor[key] = recor[key] + torch.randn_like(recor[key])
                                new_records.append(new_recor)
                            noisy_actions = actor.forward_batch(new_records)
                            temp += noisy_actions
                        expected_action = temp/5
                        
                    actions_l1 = F.softmax(actions[:, 0], dim=0)
                    expected_l1 = F.softmax(expected_action[:, 0], dim=0)
                    dist1 = Categorical(probs=actions_l1)
                    dist2 = Categorical(probs=expected_l1)
                    kl_div = kl_divergence(dist1, dist2)
                    actions_l2 = F.sigmoid(actions[:, 1])
                    expected_l2 = F.sigmoid(expected_action[:, 1])
                    distance = (actions_l2 * torch.log(actions_l2 / expected_l2) + 
                                (1 - actions_l2) * torch.log((1 - actions_l2) / (1 - expected_l2))) * expected_l1
                    distance = distance.sum() + kl_div
                    action_prob = F.sigmoid(distance)
                    with torch.no_grad():
                        value = critic(records.state)-critics(records.nstate)
                    loss = sum(torch.log(action_prob)*value)
                    actor_optimizer.zero_grad()
                    loss.backward()
                    actor_optimizer.step()
                    
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
        critic = self.critics[inter_id]
        critic_optimizer = self.critics_optimizer[inter_id]
        flag = True # 自定义的
        num_epochs = 100
        batch_size = 40
        if flag:
            for epoch in range(num_epochs):
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']), batch_size):
                    indices = permutation[i:i+batch_size]
                    temp_records = records['b_state'][indices]
                    
                    values = critic.forward_batch(temp_records)
                    with torch.no_grad():
                        temp_records = records['a_state'][indices]
                        expected_values = critic.forward_batch(temp_records)
                    expected_values = expected_values + torch.tensor(records['reward'][indices]).reshape(expected_values.shape)
                    loss = sum((expected_values-values)**2)
                    critic_optimizer.zero_grad()
                    loss.backward()
                    critic_optimizer.step()
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{Registry.mapping}")
    logging.info(f"nihao")
    True