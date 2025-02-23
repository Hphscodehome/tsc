#region other-package
import torch
from collections import defaultdict,deque
import logging
import time
import asyncio
import pdb
from torch.utils.tensorboard import SummaryWriter
import os
#endregion

#region my-package
from define_world_agent import World_agent
from datatype.define_world import World
from utils.paths import get_unique_log_dir
#endregion



class Game():
    def __init__(self, sumocfg):
        self.cfg = sumocfg
        self.world = World(self.cfg)
        self.world_agent = World_agent(self.world.inters)
        self.max_length = 20000
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        self.state = self.world.reset()
        self.train_writer = SummaryWriter(os.path.join(get_unique_log_dir(),'train'))
        self.eval_writer = SummaryWriter(os.path.join(get_unique_log_dir(),'eval'))
        self.global_step = 0
        self.eval_results = defaultdict(list)
        
    def reset(self):
        
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        self.state = self.world.reset()
        
    def play(self, end = 20,epoch=0):
        
        self.reset()
        step = 0
        while step < end:
            actions,log_probs = self.world_agent.step(self.state)
            obs, rewards, dones, infos = self.world.step(actions)
            for inter in self.world.inters:
                self.recoder[inter.id]['b_state'].append(self.state[inter.id])
                self.train_writer.add_scalar(f"reward_{str(epoch)}/{inter.id}", rewards[inter.id], step)
                self.recoder[inter.id]['reward'].append(rewards[inter.id])
                self.recoder[inter.id]['log_prob'].append(log_probs[inter.id])
                self.recoder[inter.id]['action'].append(actions[inter.id])
                self.recoder[inter.id]['a_state'].append(obs[inter.id])
                self.recoder[inter.id]['done'].append(dones[inter.id])
            self.infos.append(infos)
            for key in ['throughput','average_delay','wait_time_ascend','total_vehicles','total_wait_nums']:
                self.train_writer.add_scalar(f"infos_{str(epoch)}/{inter.id}/{key}", infos[inter.id].dict()[key], step)
            self.state = obs
            step += 1
        
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
            for key in ['throughput','average_delay','wait_time_ascend','total_vehicles']:
                self.eval_writer.add_scalar(f"infos_{str(round)}/{inter.id}/{key}", infos[inter.id].dict()[key], step)
            
            self.state = obs
            step += 1
            
        for inter in self.world.inters:
            self.eval_results[inter.id].append(total_reward[inter.id])
            self.eval_writer.add_scalar(f"result/{inter.id}", self.eval_results[inter.id][-1], len(self.eval_results[inter.id]))
        
    async def train(self):
        #pdb.set_trace()
        await self.world_agent.optimize(self.recoder)
        return True
        
async def main():
    logging.basicConfig(level=logging.INFO)
    sumocfg = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'
    game = Game(sumocfg=sumocfg)
    for i in range(20):
        game.play(end= 1000, epoch= i)
        logging.info(game.infos)
        await game.train()
        game.evaluate(end = 500, round = i)
        game.world_agent.save(round=i)
    # pdb.set_trace()
    logging.info(f"done with: {game.cfg}")
    
if __name__ == '__main__':
    #pdb.set_trace()
    asyncio.run(main())
    