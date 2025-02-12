#region other-package
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import pdb
#endregion


#region my-package
from registry.define_registry import Registry
from model.baseline import Model
#endregion




# 为每个属性定义一个感知网络
@Registry.register('actor','feature_specific')
class feature_specific_Model_actor(Model):
    def __init__(self, **kwargs):
        """
        :param use_func: list[str]
        """
        super().__init__(**kwargs)
        self.use_func = kwargs['use_func']
        self.lane_in = 2
        self.lane_out = 4
        self.vehicle_in = 4
        self.vehicle_out = 12
        self.vehicle_head = 3
        self.phase_size = 3
        self.phase_out = 4
        self.total_head = 4
        self.merge_in = 0
        for func in self.use_func:
            if 'lane' in func:
                self.merge_in += self.lane_out
            elif 'vehicle' in func:
                self.merge_in += self.vehicle_out
            else:
                self.merge_in += self.phase_out
        self.networks = nn.ModuleDict()
        assert self.merge_in%self.total_head==0,'invalid head nums'
        for key in self.use_func:
            if 'lane' in key:
                #batch * lanes * 2
                self.networks[key] = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.lane_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.lane_out)
                )])
            elif 'vehicle' in key:
                # batch * lanes * length * 2
                # (batch * lanes) * length * 2
                # length * (batch * lanes) * 2
                # 添加qkv投影
                '''
                self.networks[key].append((nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ),nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ),nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                )))
                self.networks[key].append(nn.MultiheadAttention(self.vehicle_out, self.vehicle_head))
                '''
                qkv = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ) for _ in range(3)])  # 三个网络对应Q, K, V
                self.networks[key] = nn.ModuleList([qkv, nn.MultiheadAttention(self.vehicle_out, self.vehicle_head)])
            else:
                #self.networks[key].append(nn.Embedding(self.phase_size, self.phase_out))
                self.networks[key] = nn.ModuleList([nn.Embedding(self.phase_size, self.phase_out)])
        self.merge = nn.MultiheadAttention(self.merge_in, self.total_head)
        self.output_layer = nn.Linear(self.merge_in,2)
        self.apply(self._init_weights)
        
    def forward(self,obs):
        obs = self.preprocess_obs(obs)
        emb = None
        for key in self.use_func:
            obs[key] = torch.unsqueeze(obs[key],0)
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
                embedding = embedding.transpose(0, 1)
                embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
                embedding = embedding.reshape(batch_size,lanes_size,-1)# batch * lanes * emb
            else:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            if emb == None:
                emb = embedding
            else:
                emb = torch.cat([emb, embedding], dim=-1)
        # 车道级合并
        mask = obs['mask'] #mask = np.tile(obs['mask'][np.newaxis, :, :], (batch_size*self.total_head, 1, 1))
        mask = mask.clone().detach().to(self.device).type(torch.float)
        #mask = torch.tensor(mask,dtype=torch.float).to(self.device)
        embedding, weight = self.merge(emb.transpose(0, 1),emb.transpose(0, 1),emb.transpose(0, 1),attn_mask = mask.bool())
        embedding = embedding.transpose(0, 1)
        embedding = self.output_layer(embedding)
        embedding = torch.tanh(embedding)
        action = torch.squeeze(embedding).cpu()
        return action
    
    def preprocess_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        result = defaultdict(torch.tensor)
        for key in self.use_func:
            stat = obs[key]
            if 'phase' in key:
                stat = torch.from_numpy(stat).long().to(self.device)
            else:
                stat = torch.from_numpy(stat).float().to(self.device)
            result[key] = stat
        result['mask'] = torch.from_numpy(obs['mask']).to(self.device)
        return result
    
    def preprocess_batch_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        result = defaultdict(torch.tensor)
        for key in self.use_func:
            stat = [item[key] for item in obs]
            stat = np.stack(stat)
            if 'phase' in key:
                stat = torch.from_numpy(stat).long().to(self.device)
            else:
                stat = torch.from_numpy(stat).float().to(self.device)
            result[key] = stat
        #pdb.set_trace()
        result['mask'] = torch.from_numpy(obs[-1]['mask']).to(self.device)
        return result
    
    def forward_batch(self,obs):
        #pdb.set_trace()
        obs = self.preprocess_batch_obs(obs)
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
                embedding = embedding.transpose(0, 1)
                embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
                embedding = embedding.reshape(batch_size,lanes_size,-1)# batch * lanes * emb
            else:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            if emb == None:
                emb = embedding
            else:
                emb = torch.cat([emb, embedding], dim=-1)
        # 车道级合并
        #pdb.set_trace()
        mask = obs['mask'] #mask = np.tile(obs['mask'][np.newaxis, :, :], (batch_size*self.total_head, 1, 1))
        mask = mask.clone().detach().to(self.device).type(torch.float)
        #mask = torch.tensor(mask,dtype=torch.float).to(self.device)
        embedding, weight = self.merge(emb.transpose(0, 1),emb.transpose(0, 1),emb.transpose(0, 1),attn_mask = mask.bool())
        embedding = embedding.transpose(0, 1)
        embedding = self.output_layer(embedding)
        embedding = torch.tanh(embedding)
        action = torch.squeeze(embedding).cpu()
        return action

@Registry.register('critic','feature_specific')
class feature_specific_Model_critic(Model):
    def __init__(self, **kwargs):
        """
        :param use_func: list[str]
        """
        super().__init__(**kwargs)
        self.use_func = kwargs['use_func']
        self.lane_in = 2
        self.lane_out = 4
        self.vehicle_in = 4
        self.vehicle_out = 12
        self.vehicle_head = 3
        self.phase_size = 3
        self.phase_out = 4
        self.total_head = 4
        self.merge_in = 0
        for func in self.use_func:
            if 'lane' in func:
                self.merge_in += self.lane_out
            elif 'vehicle' in func:
                self.merge_in += self.vehicle_out
            else:
                self.merge_in += self.phase_out
        self.networks = nn.ModuleDict()
        assert self.merge_in%self.total_head==0,'invalid head nums'
        for key in self.use_func:
            if 'lane' in key:
                #batch * lanes * 2
                self.networks[key] = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.lane_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.lane_out)
                )])
            elif 'vehicle' in key:
                # batch * lanes * length * 2
                # (batch * lanes) * length * 2
                # length * (batch * lanes) * 2
                # 添加qkv投影
                '''
                self.networks[key].append((nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ),nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ),nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                )))
                self.networks[key].append(nn.MultiheadAttention(self.vehicle_out, self.vehicle_head))
                '''
                qkv = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.vehicle_in, 7),
                    nn.LeakyReLU(),
                    nn.Linear(7, 11),
                    nn.LeakyReLU(),
                    nn.Linear(11, self.vehicle_out)
                ) for _ in range(3)])  # 三个网络对应Q, K, V
                self.networks[key] = nn.ModuleList([qkv, nn.MultiheadAttention(self.vehicle_out, self.vehicle_head)])
            else:
                #self.networks[key].append(nn.Embedding(self.phase_size, self.phase_out))
                self.networks[key] = nn.ModuleList([nn.Embedding(self.phase_size, self.phase_out)])
        self.merge = nn.MultiheadAttention(self.merge_in, self.total_head)
        self.output_layer = nn.Linear(self.merge_in,1)
        self.apply(self._init_weights)
        
    def forward(self,obs):
        obs = self.preprocess_obs(obs)
        emb = None
        for key in self.use_func:
            obs[key] = torch.unsqueeze(obs[key],0)
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
                embedding = embedding.transpose(0, 1)
                embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
                embedding = embedding.reshape(batch_size,lanes_size,-1)# batch * lanes * emb
            else:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            if emb == None:
                emb = embedding
            else:
                emb = torch.cat([emb, embedding], dim=-1)
        # 车道级合并
        mask = obs['mask'] #mask = np.tile(obs['mask'][np.newaxis, :, :], (batch_size*self.total_head, 1, 1))
        mask = mask.clone().detach().to(self.device).type(torch.float)
        #mask = torch.tensor(mask,dtype=torch.float).to(self.device)
        embedding, weight = self.merge(emb.transpose(0, 1),emb.transpose(0, 1),emb.transpose(0, 1),attn_mask = mask.bool())
        embedding = embedding.transpose(0, 1)
        embedding = self.output_layer(embedding)
        embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
        embedding = torch.tanh(embedding)*200
        return embedding
    
    
    def preprocess_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        result = defaultdict(torch.tensor)
        for key in self.use_func:
            stat = obs[key]
            if 'phase' in key:
                stat = torch.from_numpy(stat).long().to(self.device)
            else:
                stat = torch.from_numpy(stat).float().to(self.device)
            result[key] = stat
        result['mask'] = torch.from_numpy(obs['mask']).to(self.device)
        return result
    
    def preprocess_batch_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        result = defaultdict(torch.tensor)
        for key in self.use_func:
            stat = [item[key] for item in obs]
            stat = np.stack(stat)
            if 'phase' in key:
                stat = torch.from_numpy(stat).long().to(self.device)
            else:
                stat = torch.from_numpy(stat).float().to(self.device)
            result[key] = stat
        #pdb.set_trace()
        result['mask'] = torch.from_numpy(obs[-1]['mask']).to(self.device)
        return result
    
    def forward_batch(self,obs):
        #pdb.set_trace()
        obs = self.preprocess_batch_obs(obs)
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
                embedding = embedding.transpose(0, 1)
                embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
                embedding = embedding.reshape(batch_size,lanes_size,-1)# batch * lanes * emb
            else:
                embedding = self.networks[key][0](obs[key])# batch * lanes * emb
            if emb == None:
                emb = embedding
            else:
                emb = torch.cat([emb, embedding], dim=-1)
        # 车道级合并
        #pdb.set_trace()
        mask = obs['mask'] #mask = np.tile(obs['mask'][np.newaxis, :, :], (batch_size*self.total_head, 1, 1))
        mask = mask.clone().detach().to(self.device).type(torch.float)
        #mask = torch.tensor(mask,dtype=torch.float).to(self.device)
        embedding, weight = self.merge(emb.transpose(0, 1),emb.transpose(0, 1),emb.transpose(0, 1),attn_mask = mask.bool())
        embedding = embedding.transpose(0, 1)
        embedding = self.output_layer(embedding)
        embedding = embedding.mean(dim=1)# (batch*lanes) * emb_length
        embedding = torch.tanh(embedding)*200
        return embedding
    