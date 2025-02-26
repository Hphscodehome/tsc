#region other-package
import torch
from collections import defaultdict,deque
import logging
import time
import asyncio
import pdb
from torch.utils.tensorboard import SummaryWriter
import os,random
#endregion

#region my-package
from define_world_agent import World_agent
from datatype.define_world import World
from utils.paths import get_unique_log_dir
#endregion



class Game():
    def __init__(self, sumocfg, train=True, num_envs=10):  # 新增num_envs参数
        self.cfg = sumocfg
        self.num_envs = num_envs
        self.world = World(self.cfg, train=train)  # sumo env
        self.world_agent = World_agent(self.world.inters) # pytorch model
        self.max_length = 20000
        self.train_writer = SummaryWriter(os.path.join(get_unique_log_dir(), 'train'))
        self.eval_writer = SummaryWriter(os.path.join(get_unique_log_dir(), 'eval'))
        self.global_step = 0
        self.state = self.world.state
        self.eval_results = defaultdict(list)
        
    def reset(self):
        self.state = self.world.reset()
    
    def run_single_trajectory(self, seed, end, epoch):
        torch.manual_seed(seed)  # 确保每个进程的随机性不同
        recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        infos = deque(maxlen=self.max_length)
        state = self.world.reset()
        for step in range(end):
            actions, log_probs = self.world_agent.step(state)
            obs, rewards, dones, infos_step = self.world.step(actions)
            for inter in self.world.inters:
                inter_id = inter.id
                recoder[inter_id]['b_state'].append(state[inter_id])
                recoder[inter_id]['reward'].append(rewards[inter_id])
                recoder[inter_id]['log_prob'].append(log_probs[inter_id])
                recoder[inter_id]['action'].append(actions[inter_id])
                recoder[inter_id]['a_state'].append(obs[inter_id])
                recoder[inter_id]['done'].append(dones[inter_id])
            infos.append(infos_step)
            state = obs
        return recoder, infos
    
    def play(self, end=20, epoch=0):
        seeds = [random.randint(0, 10000) for _ in range(self.num_envs)]
        # 合并所有轨迹数据
        all_recoder = []
        all_infos = []
        for seed in seeds:
            results = self.run_single_trajectory(seed, end, epoch)
            recoder, infos = results
            all_recoder.append(recoder)
            all_infos.append(infos)
        
        # 记录日志
        for i, infos_batch in enumerate(all_infos[:1]):  # 只记录第一个环境的infos作为示例
            for inter in self.world.inters:
                inter_id = inter.id
                for step, info in enumerate(infos_batch):
                    for key in ['throughput', 'average_delay', 'wait_time_ascend', 'total_vehicles', 'total_wait_nums', 'waitnums_asc', 'vehicles_dec']:
                        self.train_writer.add_scalar(f"infos_{epoch}_{inter_id}/{key}", info[inter_id].model_dump()[key], step)
        return all_recoder,all_infos
    
    def evaluate(self,end=20,round=0):
        self.reset()
        step = 0
        total_reward = defaultdict(lambda: 0)
        while step < end:
            actions , _ = self.world_agent.eval_step(self.state)
            eva_values = self.world_agent.eval_state(self.state)
            obs, rewards, dones, infos = self.world.step(actions)
            for inter in self.world.inters:
                total_reward[inter.id] += rewards[inter.id]
                self.eval_writer.add_scalar(f"reward_{str(round)}/{inter.id}", rewards[inter.id], step)
                self.eval_writer.add_scalar(f"total_reward_{str(round)}/{inter.id}", total_reward[inter.id], step)
                self.eval_writer.add_scalar(f"eval_values_{str(round)}/{inter.id}", eva_values[inter.id], step)
            for key in ['throughput','average_delay','wait_time_ascend','total_vehicles','total_wait_nums','waitnums_asc','vehicles_dec']:
                self.eval_writer.add_scalar(f"infos_{str(round)}_{inter.id}/{key}", infos[inter.id].model_dump()[key], step)
            self.state = obs
            step += 1
        for inter in self.world.inters:
            self.eval_results[inter.id].append(total_reward[inter.id])
            self.eval_writer.add_scalar(f"result/{inter.id}", self.eval_results[inter.id][-1], len(self.eval_results[inter.id]))
            
    def random(self,end=20,round=0):
        self.reset()
        step = 0
        total_reward = defaultdict(lambda: 0)
        while step < end:
            actions , _ = self.world_agent.random_step(self.state)
            eva_values = self.world_agent.eval_state(self.state)
            obs, rewards, dones, infos = self.world.step(actions)
            for inter in self.world.inters:
                total_reward[inter.id] += rewards[inter.id]
                self.eval_writer.add_scalar(f"reward_{str(round)}/{inter.id}", rewards[inter.id], step)
                self.eval_writer.add_scalar(f"total_reward_{str(round)}/{inter.id}", total_reward[inter.id], step)
                self.eval_writer.add_scalar(f"eval_values_{str(round)}/{inter.id}", eva_values[inter.id], step)
            for key in ['throughput','average_delay','wait_time_ascend','total_vehicles','total_wait_nums','waitnums_asc']:
                self.eval_writer.add_scalar(f"infos_{str(round)}_{inter.id}/{key}", infos[inter.id].model_dump()[key], step)
            self.state = obs
            step += 1
        for inter in self.world.inters:
            self.eval_results[inter.id].append(total_reward[inter.id])
            self.eval_writer.add_scalar(f"result/{inter.id}", self.eval_results[inter.id][-1], len(self.eval_results[inter.id]))
            
    async def train(self,all_recoder):
        await self.world_agent.optimize(all_recoder)
        return True
        
async def main():
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(get_unique_log_dir()+'debug_log.log', encoding='utf-8')  # 输出到文件
        ]
    )
    sumocfg = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'
    game = Game(sumocfg=sumocfg,train=True,num_envs=10)
    best_reward = float('-inf')
    patience = 10
    patience_counter = 0
    for i in range(500):
        all_recoder , all_info = game.play(end = 1000, epoch = i)
        logging.info(all_info)
        await game.train(all_recoder)
        game.evaluate(end = 500, round = i)
        total_reward = sum(game.eval_results[inter.id][-1] for inter in game.world.inters)
        if total_reward > best_reward:
            best_reward = total_reward
            patience_counter = 0
            game.world_agent.save(round=i,exp=1)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info(f"第 {i} 个epoch时早停")
            break
    logging.info(f"done with: {game.cfg}")
    
if __name__ == '__main__':
    asyncio.run(main())
    