#region other-package
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import logging
import asyncio
import pdb
import numpy as np
import random
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
        self.actors_prob = {}
        self.target_critics = {}
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
            self.target_critics[inter.id] = Registry.mapping['critic']['feature_specific'](**kwargs)
            self.target_critics[inter.id].load_state_dict(self.critics[inter.id].state_dict())
            self.actors_optimizer[inter.id] = optim.SGD(self.actors[inter.id].parameters(), lr=0.01)
            self.critics_optimizer[inter.id] = optim.SGD(self.critics[inter.id].parameters(), lr=0.01)
            self.actors_prob[inter.id] = 0.8
            
    def step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        for inter_id in list(self.actors.keys()):
            if random.random() > self.actors_prob[inter_id]:
                with torch.no_grad():
                    actions[inter_id] = self.actors[inter_id].forward(obs[inter_id])
                #logging.info(f"1,guding,{actions[inter_id]}")
            else:
                with torch.no_grad():
                    actions[inter_id] = torch.randn_like(self.actors[inter_id].forward(obs[inter_id]))
                #logging.info(f"2,suiji,{actions[inter_id]}")
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
        """
        使用PPO优化actor模型
        records: dict containing 'b_state', 'reward', 'action', 'a_state', 'done', 'old_log_prob'
        """
        #计算该状态下对应动作出现的概率。
        #计算该状态下对应动作的优势。
        #求导优化
        #pdb.set_trace()
        actor = self.actors[inter_id]
        critic = self.critics[inter_id]
        actor_optimizer = self.actors_optimizer[inter_id]
        flag = True# 自定义的
        num_epochs = 4
        batch_size = 100
        clip_param = 0.2  # PPO剪切参数
        gamma = 0.99      # 折扣因子
        lam = 0.95        # GAE参数
        max_grad_norm = 0.5  # 梯度裁剪
        if flag:
            for epoch in range(num_epochs):
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']),batch_size):
                    logging.info(f"i,{i}")
                    with torch.autograd.detect_anomaly():
                        indices = permutation[i:i+batch_size]
                        temp_records = np.array(records['b_state'], dtype=object)[indices]
                        #pdb.set_trace()
                        actions = actor.forward_batch(temp_records)
                        logging.info(f"{actions.flatten()}")
                        #pdb.set_trace()
                        with torch.no_grad():
                            temp = 0
                            for _ in range(5):
                                new_records = []
                                for recor in temp_records:
                                    new_recor = {}
                                    for key in recor.keys():
                                        if (key != 'mask') and ('phase' not in key):
                                            new_recor[key] = recor[key] + np.random.randn(*recor[key].shape)
                                        else:
                                            new_recor[key] = recor[key]
                                    new_records.append(new_recor)
                                #pdb.set_trace()
                                noisy_actions = actor.forward_batch(new_records)
                                temp += noisy_actions
                            expected_action = temp/5
                        #pdb.set_trace()
                        logging.info(f"{expected_action.flatten()}")
                        batch_size, num_samples, num_features = actions.shape  # 这里 num_features 应该是 2
                        actions_l1 = F.softmax(actions[..., 0], dim=1)  # dim=1 因为我们要在第二个维度上应用 softmax
                        expected_l1 = F.softmax(expected_action[..., 0], dim=1)
                        kl_div = torch.sum(actions_l1 * torch.log(actions_l1 + 1e-8) - actions_l1 * torch.log(expected_l1 + 1e-8),dim=1)
                        
                        actions_l2 = F.sigmoid(actions[..., 1])
                        expected_l2 = F.sigmoid(expected_action[..., 1])
                        distance = (actions_l2 * torch.log(actions_l2 + 1e-8) - actions_l2 * torch.log(expected_l2 + 1e-8) + \
                                    (1 - actions_l2) * torch.log(1 - actions_l2 + 1e-8) - (1 - actions_l2) * torch.log(1 - expected_l2 + 1e-8)) * expected_l1
                        distance = distance.sum(dim=1) + kl_div
                        action_prob = F.sigmoid(distance)
                        logging.info(f"action_prob : {action_prob.flatten()}")
                        with torch.no_grad():
                            values = critic.forward_batch(np.array(records['b_state'], dtype=object)[indices]).squeeze()
                            next_values = critic.forward_batch(np.array(records['a_state'], dtype=object)[indices]).squeeze()
                            deltas = torch.from_numpy(np.array(records['reward'])[indices]).float().reshape(next_values.shape) + gamma * next_values - values
                            advantages = torch.zeros_like(deltas)
                            advantages[-1] = deltas[-1]
                            for t in reversed(range(len(deltas) - 1)):
                                advantages[t] = deltas[t] + gamma * lam * advantages[t + 1]
                            returns = advantages + values
                            value = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                            #value = critic.forward_batch(np.array(records['b_state'], dtype=object)[indices])
                        logging.info(f"value : {value.flatten()}")
                        loss = sum(torch.log(action_prob+1e-8)*value.flatten())
                        logging.info(f"{loss},actor,{[key for key in actor.output_layer.parameters()]}")
                        actor_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        actor_optimizer.step()
                        logging.info(f"{loss},actor,{[key for key in actor.output_layer.parameters()]}")
        #pdb.set_trace()
        return True
                    
    async def optimize_critic(self, inter_id, records):
        """
        使用改进的DQN训练Q网络
        records: dict containing 'b_state', 'reward', 'action', 'a_state', 'done'
        """
        #计算该状态下对应状态的状态价值
        #计算下个状态的状态价值
        #根据reward调整状态价值函数
        #pdb.set_trace()
        critic = self.critics[inter_id]  # 在线网络
        target_critic = self.target_critics[inter_id]  # 目标网络
        critic_optimizer = self.critics_optimizer[inter_id]
        flag = True # 自定义的
        batch_size = 100
        gamma = 0.99  # 折扣因子
        num_epochs = 5
        clip_grad_norm = 1.0  # 梯度裁剪
        if flag:
            for epoch in range(num_epochs):
                #pdb.set_trace()
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']), batch_size):
                    #logging.info(f"i:{i}")
                    indices = permutation[i:i+batch_size]
                    temp_records = np.array(records['b_state'], dtype=object)[indices]
                    #pdb.set_trace()
                    values = critic.forward_batch(temp_records)
                    #logging.info(f"{values.flatten()}")
                    with torch.no_grad():
                        temp_records = np.array(records['a_state'], dtype=object)[indices]
                        expected_values = target_critic.forward_batch(temp_records)
                    #pdb.set_trace()
                    expected_values = expected_values + gamma * torch.from_numpy(np.array(records['reward'])[indices]).float().reshape(expected_values.shape)
                    expected_values = torch.clamp(expected_values, -199, 199)
                    #logging.info(f"{expected_values.flatten()}")
                    loss = sum((expected_values-values)**2)/batch_size
                    logging.info(f"{loss.item():.4f},critic,{[key for key in critic.output_layer.parameters()]}")
                    critic_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_grad_norm)
                    critic_optimizer.step()
                    logging.info(f"{loss.item():.4f},critic,{[key for key in critic.output_layer.parameters()]}")
            target_critic.load_state_dict(critic.state_dict())
            logging.info("Target network updated")
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{Registry.mapping}")
    logging.info(f"nihao")
    True