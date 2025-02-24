#region other-package
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from collections import defaultdict
import logging
import asyncio
import numpy as np
import random
import os
#endregion

#region my-package
from model.define_modelv2 import * 
from registry.define_registry import Registry
from utils.constants import obs_fn
from utils.paths import get_unique_log_dir
#endregion

class World_agent():
    def __init__(self,intersections):
        self.actors = {}
        self.target_actors = {}
        self.critics = {}
        self.target_critics = {}
        self.actors_optimizer = {}
        self.critics_optimizer = {}
        self.actors_prob = {}
        for inter in intersections:
            kwargs = {
                'use_func': obs_fn,
                'model_type': 'actor',
                'device': 'cpu',
                'log_dir': os.path.join(get_unique_log_dir(),inter.id,'actor')
            }
            self.actors[inter.id] = Registry.mapping['actor']['feature_specific'](**kwargs)
            self.target_actors[inter.id] = Registry.mapping['actor']['feature_specific'](**kwargs)
            self.target_actors[inter.id].load_state_dict(self.actors[inter.id].state_dict())
            
            kwargs = {
                'use_func': obs_fn,
                'model_type': 'critic',
                'device': 'cpu',
                'log_dir': os.path.join(get_unique_log_dir(),inter.id,'critic')
            }
            self.critics[inter.id] = Registry.mapping['critic']['feature_specific'](**kwargs)
            self.target_critics[inter.id] = Registry.mapping['critic']['feature_specific'](**kwargs)
            self.target_critics[inter.id].load_state_dict(self.critics[inter.id].state_dict())
            
            self.actors_optimizer[inter.id] = optim.Adam(self.actors[inter.id].parameters(), lr=1e-4)
            self.critics_optimizer[inter.id] = optim.Adam(self.critics[inter.id].parameters(), lr=1e-3)
            self.actors_prob[inter.id] = 0.6
            
    def save(self,round=0):
        for inter_id in list(self.actors.keys()):
            critic = self.critics[inter_id]
            actor = self.actors[inter_id]
            torch.save(critic.state_dict(), f'./pths/{inter_id}_critic_round{str(round)}_model_weights.pth')
            torch.save(actor.state_dict(), f'./pths/{inter_id}_actor_round{str(round)}_model_weights.pth')
            self.actors_prob[inter_id] = self.actors_prob[inter_id]*0.8
            
    def eval_state(self,obs):
        eval_states = {}
        for inter_id in list(self.actors.keys()):
            critic = self.critics[inter_id]
            state = obs[inter_id]
            with torch.no_grad():
                eval_states[inter_id] = critic.forward(state).flatten().item()
        return eval_states
            
    def eval_step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        for inter_id in list(self.actors.keys()):
            with torch.no_grad():
                actions[inter_id] , _ = self.actors[inter_id].get_mu_sigma(obs[inter_id])
        return actions , _
    
    def step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        log_probs = {}
        for inter_id in list(self.actors.keys()):
            if random.random() > self.actors_prob[inter_id]:
                with torch.no_grad():
                    actions[inter_id] , log_probs[inter_id] = self.actors[inter_id].forward(obs[inter_id])
            else:
                with torch.no_grad():
                    mu,sigma = self.actors[inter_id].get_mu_sigma(obs[inter_id])
                    dist = D.Normal(mu, sigma)
                    actions[inter_id] = torch.randn_like(mu)
                    log_prob = dist.log_prob(actions[inter_id]).sum()
                    log_probs[inter_id] = log_prob
        #logging.info(f"actions:{actions},log:{log_probs}")
        return actions , log_probs
    
    async def optimize(self, records):
        tasks = [self.optimize_inter(inter_id, records[inter_id]) for inter_id in self.actors.keys()]
        await asyncio.gather(*tasks)
        return True
            
    async def optimize_inter(self, inter_id, records):
        await self.optimize_inone(inter_id, records)
        return True
    
    async def optimize_inone(self,inter_id,records):
        actor = self.actors[inter_id]
        critic = self.critics[inter_id]
        actor_optimizer = self.actors_optimizer[inter_id]
        critic_optimizer = self.critics_optimizer[inter_id]
        
        def compute_gae(critic, trajectory, gamma=0.99, lam=0.95):
            with torch.no_grad():
                state_records = np.array(trajectory['b_state'], dtype=object)
                values = critic.forward_batch(state_records)
                state_records = np.array(trajectory['a_state'], dtype=object)[-1:]
                next_value = critic.forward_batch(state_records).item()
                
            rewards = np.array(trajectory["reward"])
            gae = 0
            returns = []
            
            for step in reversed(range(len(rewards))):
                if step == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = values[step+1].item()
                delta = rewards[step] + gamma * next_val- values[step].item()
                gae = delta + gamma * lam  * gae
                returns.insert(0, gae + values[step].item())
            return np.array(returns)
        
        returns = compute_gae(critic, records, gamma=0.99, lam=0.95)
        returns = torch.from_numpy(returns).float()
        logging.info(f"returns:{returns.flatten()[:10]}")
        
        with torch.no_grad():
            state_records = np.array(records['b_state'], dtype=object)
            values = critic.forward_batch(state_records)
            advantages = returns - values.flatten()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logging.info(f"advantages:{advantages.flatten()[:10]}")
        
        old_log_probs = torch.FloatTensor(records["log_prob"])
        
        flag = True
        num_epochs = 10
        batch_size = 300
        clip_param = 0.2  # PPO剪切参数
        max_grad_norm = 1.0
        
        if flag:
            for epoch in range(num_epochs):
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']),batch_size):
                    indices = permutation[i:i+batch_size]
                    
                    batch_states = np.array(records['b_state'], dtype=object)[indices]
                    mu,sigma = actor.get_mu_sigma_batch(batch_states)
                    batch_actions = np.array(records['action'])[indices]
                    batch_actions = torch.from_numpy(batch_actions).float()
                    dist = D.Normal(mu, sigma)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=[1,2])
                    logging.info(f"new_log_probs: {new_log_probs.flatten()[:10]}")
                    batch_old_log_probs = old_log_probs[indices]
                    logging.info(f"batch_old_log_probs: {batch_old_log_probs.flatten()[:10]}")
                    
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    logging.info(f"logs differ: {(new_log_probs - batch_old_log_probs).flatten()[:10]}")
                    logging.info(f"ratio: {ratio.flatten()[:10]}")
                    batch_advantages = advantages[indices]
                    logging.info(f"batch advantage: {batch_advantages.flatten()[:10]}")
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, min=1 - clip_param, max=1 + clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    #logging.info(f"actor:{actor.mu_head.weight}")
                    actor_optimizer.step()
                    logging.info(f"{policy_loss},actor:{actor.mu_head.weight.flatten()[:10]}")
                    actor.trained_step += 1
                    actor.writer.add_scalar("actor_Loss/train", policy_loss.item(), actor.trained_step)
                    # 价值函数损失（均方误差）
                    batch_returns = returns[indices]
                    logging.info(f"batch returns: {batch_returns.flatten()[:10]}")
                    value_loss = nn.MSELoss()(critic.forward_batch(np.array(records['b_state'], dtype=object)[indices]).squeeze(), batch_returns)
                    critic_optimizer.zero_grad()
                    value_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    critic_optimizer.step()
                    critic.trained_step += 1
                    critic.writer.add_scalar("critic_Loss/train", value_loss.item(), critic.trained_step)
                    logging.info(f"{value_loss},critic:{critic.value_head.weight.flatten()[:10]}")
        return True
    
    
    async def optimize_actor(self, inter_id, records):
        """
        使用PPO优化actor模型
        records: dict containing 'b_state', 'reward', 'action', 'a_state', 'done', 'old_log_prob'
        """
        #计算该状态下对应动作出现的概率。
        #计算该状态下对应动作的优势。
        #求导优化
        actor = self.actors[inter_id]
        critic = self.critics[inter_id]
        actor_optimizer = self.actors_optimizer[inter_id]
        def compute_gae(rewards, values, next_values, gamma, lam):
            deltas = rewards + gamma * next_values - values
            gae = torch.zeros_like(deltas)
            for t in reversed(range(len(deltas))):
                gae[t] = deltas[t] + gamma * lam * gae[t + 1] if t + 1 < len(deltas) else deltas[t]
            return gae
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
                        
                        actions,_ = actor.forward_batch(temp_records)
                        logging.info(f"{actions.flatten()}")
                        
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
                                
                                noisy_actions,_ = actor.forward_batch(new_records)
                                temp += noisy_actions
                            expected_action = temp/5
                        
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
                        logging.info(f"{loss},actor,{[key for key in actor.mu_head.parameters()]}")
                        actor_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        actor_optimizer.step()
                        logging.info(f"{loss},actor,{[key for key in actor.mu_head.parameters()]}")
        
        return True
                    
    async def optimize_critic(self, inter_id, records):
        """
        使用改进的DQN训练Q网络
        records: dict containing 'b_state', 'reward', 'action', 'a_state', 'done'
        """
        #计算该状态下对应状态的状态价值
        #计算下个状态的状态价值
        #根据reward调整状态价值函数
        
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
                
                permutation = torch.randperm(len(records['b_state']))
                for i in range(0,len(records['b_state']), batch_size):
                    #logging.info(f"i:{i}")
                    indices = permutation[i:i+batch_size]
                    temp_records = np.array(records['b_state'], dtype=object)[indices]
                    
                    values = critic.forward_batch(temp_records)
                    #logging.info(f"{values.flatten()}")
                    with torch.no_grad():
                        temp_records = np.array(records['a_state'], dtype=object)[indices]
                        expected_values = target_critic.forward_batch(temp_records)
                    
                    expected_values = gamma * expected_values + torch.from_numpy(np.array(records['reward'])[indices]).float().reshape(expected_values.shape)
                    expected_values = torch.clamp(expected_values, -199, 199)
                    #logging.info(f"{expected_values.flatten()}")
                    loss = sum((expected_values-values)**2)/batch_size
                    logging.info(f"{loss.item():.4f},critic,{[key for key in critic.value_head.parameters()]}")
                    critic_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_grad_norm)
                    critic_optimizer.step()
                    logging.info(f"{loss.item():.4f},critic,{[key for key in critic.value_head.parameters()]}")
            target_critic.load_state_dict(critic.state_dict())
            logging.info("Target network updated")
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{Registry.mapping}")
    logging.info(f"nihao")
    True