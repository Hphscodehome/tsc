#region other-package
import torch
from collections import defaultdict,deque
import logging
import time
#endregion

#region my-package
from define_world_agent import World_agent
from datatype.define_world import World
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
        
    def reset(self):
        self.recoder = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_length)))
        self.infos = deque(maxlen=self.max_length)
        self.state = self.world.reset()
        
    def play(self, end = 20):
        step = 0
        while step < end:
            actions = self.world_agent.step(self.state)
            obs, rewards, dones, infos = self.world.step(actions)
            for inter in self.world.inters:
                self.recoder[inter.id]['b_state'].append(self.state[inter.id])
                self.recoder[inter.id]['reward'].append(rewards[inter.id])
                self.recoder[inter.id]['action'].append(actions[inter.id])
                self.recoder[inter.id]['a_state'].append(obs[inter.id])
                self.recoder[inter.id]['done'].append(dones[inter.id])
            self.infos.append(infos)
            self.state = obs
            step += 1
        logging.info(f"Recoder:\n{self.recoder}")
        logging.info(f"Infos:\n{self.infos}")
        
    def train_actor(self,batch_record):
        for actor_id in self.world_agent.actors.keys():
            actor = self.world_agent.actors[actor_id]
            batch_obs = batch_record[actor_id]
            actor.optimize(batch_obs)
        pass
    def train_critic(self,batch_record):
        
        pass
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sumocfg = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'
    game = Game(sumocfg=sumocfg)
    game.play()
    True
    