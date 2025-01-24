#region other-package
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
#endregion


class Model(nn.Module):
    def __init__(self, **kwargs):
        """base class
        :param model_type: str, actor(output is probability), critic(output is max action value as state value), defaults to None(output is all action value)
        :param device: defaults to 'cpu'
        :param log_dir: defaults to './log/'
        """
        super().__init__()
        print(kwargs)
        model_type, device, log_dir = kwargs['model_type'], kwargs['device'], kwargs['log_dir'],
        assert model_type in [None, "actor", "critic"]
        self.model_type = model_type
        self.device = device
        self.writer = None
        self.step = 0
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        

    def _init_weights(self, module, gain=1.0):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
                
    def preprocess_obs(self, obs):
        raise NotImplementedError

    def get_out(self, out):
        """post process out by model type, actor / critic / None
        :param out: model output
        :return: 
        """
        if self.model_type == "actor":
            out = F.softmax(out, dim=-1)
        elif self.model_type == "critic":
            out = torch.mean(out, dim=-1)
        return out

class MLPModel(Model):
    def __init__(self, **kwargs):
        """use simple mlp layer to learn policy and value
        :param input_dim: int, input dimension, concat of features
        :param output_dim: int, num action
        :param model_type: str, actor, critic, None
        :param device: str
        """
        super().__init__(**kwargs)
        self.input_dim = kwargs['input_dim']
        self.output_dim = kwargs['output_dim']
        self.model = nn.Sequential(
                                    nn.Linear(self.input_dim, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 20),
                                    nn.LeakyReLU(),
                                    nn.Linear(20, self.output_dim)
                                )

        
    def forward(self, obs):
        stat, phase = self.preprocess_obs(obs)
        state = torch.cat((stat, phase), dim=-1)
        out = self.model(state)
        out = self.get_out(out)
        return out
    
    def preprocess_obs(self, obs):
        """preprocess observation to tensor
        :param obs: observation
        """
        origin_stat = obs['vehicle_map']
        origin_phase = obs['current_phase']
        origin_musk = obs['musk']
        stat = torch.from_numpy(origin_stat).float().to(self.device)
        stat = torch.flatten(stat, 1,-1)
        phase = torch.from_numpy(origin_phase).long().to(self.device)
        return stat, phase

    
class FRAPModel(Model):
    def __init__(self, input_dim, output_dim, **kargs):
        super().__init__(**kargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = 16
        # encoder 
        self.linear1 = nn.Linear(self.input_dim, 12)
        self.activation1 = nn.LeakyReLU()
        self.embedding = nn.Embedding(4, 4)
        self.encoder_out_layer = nn.Sequential(nn.Linear(16, 32), nn.LeakyReLU(), nn.Linear(32, self.embedding_dim))

        # phase competition
        self.state_conv1 = nn.Conv2d(2 * self.embedding_dim, 20, kernel_size=1)
        self.state_relu1 = nn.LeakyReLU()
        self.output_layer = nn.Sequential(
                                            nn.Conv2d(20, 20, kernel_size=1), 
                                            nn.LeakyReLU(),
                                            nn.Conv2d(20, 1, kernel_size=1)
                                        )
        self.mask = None # phase mask

    def forward(self, obs):
        stat, phase = self.preprocess_obs(obs)
        batch_size = stat.shape[0]
        num_phase = self.mask.shape[0]
        # embedding_statistics: (batch_size x num_lanes x 16)
        embedding_statistics = self.activation1(self.linear1(stat))
        embedding_phase = self.embedding(phase)
        embedding = torch.cat((embedding_statistics, embedding_phase), dim=-1)

        # embedding: (batch_size x num_lanes x output_dim)
        embedding = self.encoder_out_layer(embedding)

        # out:(batch_size x num_phases x output_dim)
        # the sum of embedding in phase passable lanes
        encoder_out = torch.zeros((batch_size, num_phase, self.embedding_dim)).to(self.device)
        for i in range(num_phase):
            encoder_out[:, i] = torch.sum(embedding[:, self.mask[i] > 0], dim=1)

        # cat every two phases
        x = [torch.stack(
            [torch.cat((encoder_out[:, i], encoder_out[:, j]), dim=-1) for j in range(num_phase) if j != i], dim=-1)
            for i in range(num_phase)]

        # x0: (batch_size x 32 x num_phases x num_phases -1)
        x0 = torch.stack(x, dim=2)
        x1 = self.state_relu1(self.state_conv1(x0))

        # out: (batch_size x num_phases)
        out = self.output_layer(x1)
        out = torch.sum(out, dim=-1)
        out = out.squeeze(1)

        out = self.get_out(out)

        return out


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