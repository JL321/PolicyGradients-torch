import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Normal 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):


    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3mu = nn.Linear(256, action_space)
        self.fc3std = nn.Linear(256, action_space)


    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        mu = self.fc3mu(z)
        std = self.fc3std(z)
        return mu, std

class QCritic(nn.Module):


    def __init__(self, state_space, action_space):
        super(QCritic, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc2a = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(256, 1)
    
    
    def forward(self, state, action):
        z = F.relu(self.fc1(state))
        z = self.fc2(z)
        az = self.fc2a(action)
        comb = F.relu(z+az)
        z = self.fc3(comb)
        return z


class SAC:

    def __init__(self, state_space, action_space, hp=None, name='SAC'): 
        self.name = name
        if hp is None:
            self.tau = 0.005
            self.lr = 3*1e-4
            self.batch_size = 256
            self.gamma = 0.99
        else:
            self.tau = hp['tau'] 
            self.lr = hp['lr'] 
            self.batch_size = hp['batch_size']
            self.gamma = hp['gamma']

        self.actor = Actor(state_space, action_space).to(device)
        self.qcritic = QCritic(state_space, action_space).to(device)
        self.qcritic2 = QCritic(state_space, action_space).to(device)
        self.tcritic = QCritic(state_space, action_space).to(device)
        self.tcritic2 = QCritic(state_space, action_space).to(device)i
        self.alpha = torch.zeros(1, device=device, requires_grad=True)
        self.target_ent = -action_space

        self.act_opt = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr).to(device)
        self.qc_opt = torch.optim.Adam(params=self.qcritic.parameters(), lr=self.lr).to(device)
        self.qc_opt2 = torch.optim.Adam(params=self.qcritic2.parameters(), lr=self.lr).to(device)
        self.alpha_opt = torch.optim.Adam(self.alpha, lr=self.lr).to(device)


    def predict(self, x, sample_size=1, detach=True):
        if detach:
            mu, std = self.actor(x).detach()
        else:
            mu, std = self.actor(x)
        act_dist = Normal(torch.squeeze(mu), torch.squeeze(std)).to(device)
        action = act_dist.sample(sample_size)
        log_prob = act_dist.log_prob(action)
        return action.numpy(), log_prob.numpy()


    def _soft_update(self):
        for q1, q1t in zip(self.qcritic.parameters(), self.tcritic.parameters()):
            q1t.data *= (1-self.tau)
            q1t.data += (self.tau)*q1.data
        for q2, q2t in zip(self.qcritic2.parameters(), self.tcritic2.parameters())
            q2t.data *= (1-self.tau)
            q2t.data += self.tau*q2.data


    def train(self, replay_buffer):
        state_set, action_set, reward_set, nstate_set, logprob_set, done_set = replay_buffer.sample()
        state_set = torch.from_numpy(state_set).to(device)
        action_set = torch.from_numpy(action_set).to(device)
        reward_set = torch.from_numpy(reward_set).to(device)
        nstate_set = torch.from_numpy(nstate_set).to(device)
        done_set = torch.from_numpy(done_set).to(device)
        logprob_set = torch.from_numpy(logprob_set).to(device)

        self.qc_opt.zero_grad()
        self.qc_opt2.zero_grad()
        act, log_prob = self.predict(nstate_set, self.batch_size)),
        qOut = torch.min(self.tcritic(nstate_set, act).detach(), self.tcritic2(nstate_set, act).detach())
        qOut -= self.alpha.detach()*(log_prob)
        target = reward_set+self.gamma*qOut
        q1loss = F.mse_loss(target, self.qcritic(state_set, action_set))
        q2loss = F.mse_loss(target, self.qcritic2(state_set, action_set))
        q1loss.backward()
        self.qc_opt.step()
        q2loss.backward()
        self.qc_opt2.step()

        self.act_opt.zero_grad()
        min_q = torch.min(self.qcritic(state_set, action_set), self.qcritic2(state_set, act))
        act, log_prob  = self.predict(state_set, self.batch_size, False)
        actor_loss = self.alpha.detach()*log_prob-min_q
        actor_loss.backward()
        self.act_opt.step()

    def save(self, path=None):
        if path is None:
            path = 'models/{}'.format(self.name)
        torch.save({
            'qcritic': self.qcritic.state_dict(),
            'qcritic2': self.qcritic2.state_dict(),
            'tcritic': self.tcritic.state_dict(),
            'tcritic2': self.tcritic.state_dict(),
            'alpha': self.alpha,
            'qopt': self.qc_opt.state_dict(),
            'qopt2': self.qc_opt2.state_dict(),
            'alphaOpt', self.alpha_opt.state_dict()
            }, path) 
    

    def load(self, path=None):
        if path is None:
            path = 'models/{}'.format(self.name)
        load_dict = torch.load(path)
        self.qcritic.load_state_dict(load_dict['qcritic'])
        self.qcritic2.load_state_dict(load_dict['qcritic2'])
        self.tcritic.load_state_dict(load_dict['tcritic'])
        self.tcritic2.load_state_dict(load_dict['tcritic2'])
        self.alpha = load_dict['alpha']
        self.qc_opt.load_state_dict(load_dict['qopt'])
        self.qc_opt2.load_state_dict(load_dict['qopt2'])
        self.alpha_opt.load_state_dict(load_dict['alphaOpt'])

