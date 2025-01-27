#region other-package
import torch
from collections import defaultdict
import logging
#endregion

#region my-package
from model.define_model import feature_specific_Model
from registry.define_registry import Registry
from utils.constants import obs_fn
#endregion

class World_agent():
    def __init__(self,intersections):
        self.actors = {}
        for inter in intersections:
            kwargs = {
                'use_func': obs_fn,
                'model_type': 'actor',
                'device': 'cpu',
                'log_dir': './logs'
            }
            self.actors[inter.id] = Registry.mapping['actor']['feature_specific'](**kwargs)
    def step(self,obs):
        actions = defaultdict(lambda: torch.tensor([]))
        for inter_id in list(self.actors.keys()):
            actions[inter_id] = self.actors[inter_id](obs[inter_id])
        return actions
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"{Registry.mapping}")
    logging.info(f"nihao")
    True