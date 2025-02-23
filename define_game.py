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
        self.writer = SummaryWriter(os.path.join(get_unique_log_dir(),'game'))
        self.global_step = 0
        
    def reset(self):
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        self.state = self.world.reset()
        
    def play(self, end = 20):
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        step = 0
        while step < end:
            actions,log_probs = self.world_agent.step(self.state)
            obs, rewards, dones, infos = self.world.step(actions)
            for inter in self.world.inters:
                self.recoder[inter.id]['b_state'].append(self.state[inter.id])
                self.writer.add_scalar(f"reward/{inter.id}", rewards[inter.id], self.global_step)
                self.recoder[inter.id]['reward'].append(rewards[inter.id])
                self.recoder[inter.id]['log_prob'].append(log_probs[inter.id])
                self.recoder[inter.id]['action'].append(actions[inter.id])
                self.recoder[inter.id]['a_state'].append(obs[inter.id])
                self.recoder[inter.id]['done'].append(dones[inter.id])
            self.infos.append(infos)
            for key in ['throughput','average_delay','wait_time_ascend','total_vehicles']:
                self.writer.add_scalar(f"infos/{inter.id}/{key}", infos[inter.id][key], self.global_step)
            self.state = obs
            step += 1
            self.global_step += 1
        #logging.info(f"Recoder:\n{self.recoder}")
        #logging.info(f"Infos:\n{self.infos}")
        
    async def train(self):
        #pdb.set_trace()
        await self.world_agent.optimize(self.recoder)
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        return True
        
async def main():
    logging.basicConfig(level=logging.INFO)
    sumocfg = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'
    game = Game(sumocfg=sumocfg)
    game.play(end=1000)
    logging.info(game.infos)
    await game.train()
    game.play(end=1000)
    logging.info(game.infos)
    await game.train()
    # pdb.set_trace()
    logging.info(f"done with: {game.cfg}")
    
if __name__ == '__main__':
    #pdb.set_trace()
    asyncio.run(main())
    