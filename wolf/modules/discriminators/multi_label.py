__author__ = 'fengyu'

from typing import Dict
from overrides import overrides
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli

from wolf.modules.discriminators.discriminator import Discriminator

class MultiLabelDiscriminator(Discriminator):
    def __init__(self, num_events, dim, activation='relu', probs=None, logits=None):
        super(MultiLabelDiscriminator, self).__init__()
        if probs is not None and logits is not None:
            raise ValueError("Either `probs` or `logits` can be specified, but not both.")
        
        if probs is not None:
            assert len(probs) == num_events, 'number of probs must match number of events.'
            probs = torch.tensor(probs).float()
            self.cat_dist = Bernoulli(probs=probs)
        elif logits is not None:
            assert len(logits) == num_events, 'number of logits must match number of events.'
            logits = torch.tensor(logits).float()
            self.cat_dist = Categorical(logits=logits)
        else:
            probs = torch.full((num_events+1, ), 0.5).float()
            self.cat_dist = Bernoulli(probs=probs)

        if activation == 'relu':
            Actv = nn.ReLU(inplace=True)
        elif activation == 'elu':
            Actv = nn.ELU(inplace=True)
        else:
            Actv = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        

        self.dim = dim
        self.embed = nn.Embedding(num_events+1, dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            Actv,
            nn.Linear(4 * dim, 4 * dim),
            Actv,
            nn.Linear(4 * dim, dim)
        )

        self.register_buffer('range_list', Variable(torch.LongTensor(list(range(num_events+1))), requires_grad=False))

        self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    @overrides
    def to_device(self, device):
        logits = self.cat_dist.logits.to(device)
        self.cat_dist = Bernoulli(logits=logits)
        self.range_list.to(device)

    @overrides
    def init(self, x, y=None, init_scale=1.0):
        with torch.no_grad():
            z, KL = self.sampling_and_KL(x, y=y)
            return z.squeeze(1), KL

    @overrides
    def sample_from_prior(self, nsamples=1, device=torch.device('cpu')):
        # [nsamples, num_events+1]
        cids = self.cat_dist.sample((nsamples, )).to(device)
        
        # [nsamples, num_events+1]
        cids = self.range_list * cids

        # [nsamples, num_events+1, dim] => [nsamples, dim] => [nsamples, dim]
        return self.net(self.embed(cids).sum(1))
        
    @overrides
    def sample_from_posterior(self, x, y=None, nsamples=1, random=True):
        assert y is not None
        # [batch, nsamples]
        log_probs = x.new_zeros(x.size(0), nsamples)

        # y: [batch, num_events+1] => [batch, num_events+1, dim] => [batch, dim] => [batch, dim]
        z = self.net(self.embed(self.range_list * y).sum(1)).unsqueeze(1) + log_probs.unsqueeze(2)
        return z, log_probs
    
    @overrides
    def sampling_and_KL(self, x, y=None, nsamples=1):
        # print('x shape is ', x.shape)
        # print('y shape is ', y.shape)
        z, _ = self.sample_from_posterior(x, y=y, nsamples=nsamples, random=True)
        logits = self.cat_dist.logits
        log_probs_prior = self.cat_dist.log_prob(y.float()).sum()
        KL = -log_probs_prior
        # print('z shape is ', z.shape)
        return z, KL
    
    @classmethod
    def from_params(cls, params: Dict) -> "MultiLabelDiscriminator":
        return MultiLabelDiscriminator(**params)
    
MultiLabelDiscriminator.register('multilabel')
        