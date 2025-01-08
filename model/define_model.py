#region other-package
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
#endregion


#region my-package
from registry.define_registry import Registry
from model.baseline import Model
#endregion




# 为每个属性定义一个感知网络
@Registry.register('model','feature_specific')
class AllattrModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(kwargs)
        self.use_func = kwargs['use_func']
        self.networks = defaultdict(list)
        for key in self.use_func:
            if 'lane' in key:
                #batch * lanes * 2
                self.networks[key] = [nn.Sequential(
                    nn.Linear(2, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, 4)
                )]
            elif 'vehicle' in key:
                # batch * lanes * length * 2
                # (batch * lanes) * length * 2
                # length * (batch * lanes) * 2
                # 添加qkv投影
                self.networks[key].append((nn.Sequential(
                    nn.Linear(2, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, 12)
                ),nn.Sequential(
                    nn.Linear(2, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, 12)
                ),nn.Sequential(
                    nn.Linear(2, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, 12)
                )))
                self.networks[key].append(nn.MultiheadAttention(12, 3))
            else:
                self.networks[key].append(nn.Embedding(4, 4))
        self.merge = nn.MultiheadAttention(12, 3)
        self.output_layer = nn.Linear(12,1)
    def forward(self,obs):
        emb = None
        for key in self.use_func:
            batch_size = len(obs[key])
            lanes_size = len(obs[key][0])
            if 'lane' in key:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            elif 'vehicle' in key:
                # batch*lanes*length*2
                query_net, key_net, value_net = self.networks[key][0]
                querys = query_net(obs[key])
                keys = key_net(obs[key])
                values = value_net(obs[key])
                querys = querys.reshape(-1,*querys.shape[2:])
                keys = keys.reshape(-1,*keys.shape[2:])
                values = values.reshape(-1,*values.shape[2:])
                querys = querys.transpose(0, 1)
                keys = keys.transpose(0, 1)
                values = values.transpose(0, 1)
                embedding, embedding_weight = self.networks[key][1](querys,keys,values)
                embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
                embedding = embedding.reshape(batch_size,lanes_size,-1)# batch * lanes * emb
            else:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            if emb == None:
                emb = embedding
            else:
                emb = torch.cat([emb, embedding], dim=-1)
        # 车道级合并
        mask = np.tile(obs['mask'][np.newaxis, :, :], (batch_size*3, 1, 1))
        embedding, weight = self.merge(emb.transpose(0, 1),emb.transpose(0, 1),emb.transpose(0, 1),attn_mask = mask)
        embedding = embedding.reshape(batch_size,lanes_size,-1)
        return self.output_layer(embedding)
    
    def preprocess_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        origin_stat = [item['vehicle_map'] for item in obs]
        origin_phase = [item['current_phase'] for item in obs]
        mask = obs
        self.mask = mask
        # (batch_size, num_lane, num_feature)
        stat = torch.from_numpy(origin_stat).float().to(self.device)
        phase = torch.from_numpy(origin_phase).long().to(self.device)
        return stat, phase