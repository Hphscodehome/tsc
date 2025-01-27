from datatype.define_world import World
from collections import defaultdict
import torch
import logging
logging.basicConfig(level=logging.INFO)
sumocfg = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'
world = World(sumocfg)
step = 0
while step < 20:
    actions = defaultdict(lambda : torch.tensor([]))
    for inter in world.inters:
        actions[inter.id] = torch.randn(len(inter.traffic_light_lanes),2)
    logging.info(f"actions:{actions}")
    obs, rewards, dones, infos = world.step(actions)
    logging.info(f"obs, rewards, dones, infos : {obs},\n,{rewards},\n,{dones},\n,{infos}")
    step += 1
world.close()