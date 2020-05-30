import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal 
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 1e-6
min_std = -20
max_std = 2

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
        std = torch.clamp(std, min_std, max_std)
        return mu, std

class QCritic(nn.Module):


    def __init__(self, state_space, action_space):
        super(QCritic, self).__init__()
        self.fc1 = nn.Linear(state_space+action_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    
    def forward(self, state, action):
        z = torch.cat([state, action], 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z


class SAC:

    def __init__(self, state_space, action_space, hp=None, name='SAC'): 
        self.name = name
        if hp is None:
            self.tau = 0.005
            self.lr = 3*(1e-4)
            self.batch_size = 256
            self.gamma = 0.99
            self.max_action = 1
            self.min_action = 0
        else:
            self.tau = hp['tau'] 
            self.lr = hp['lr'] 
            self.batch_size = hp['batch_size']
            self.gamma = hp['gamma']
            self.max_action = hp['max_action']
            self.min_action = hp['min_action']
        self.actor = Actor(state_space, action_space).to(device)
        self.qcritic = QCritic(state_space, action_space).to(device)
        self.qcritic2 = QCritic(state_space, action_space).to(device)
        self.tcritic = QCritic(state_space, action_space).to(device)
        self.tcritic2 = QCritic(state_space, action_space).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_ent = -action_space
        print("Target Entropy: {}".format(self.target_ent))
        print("Verify Device: {}".format(device))
        self.act_opt = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)
        self.qc_opt = torch.optim.Adam(params=self.qcritic.parameters(), lr=self.lr)
        self.qc_opt2 = torch.optim.Adam(params=self.qcritic2.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.action_scale = torch.tensor((self.max_action-self.min_action)/2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((self.max_action+self.min_action)/2., dtype=torch.float32).to(device)

    def predict(self, x, pred=True, internalCall=False, test=False):
        if test:
            self.actor.eval()
        if pred and not internalCall:
            x = torch.from_numpy(x).float().to(device)
        if pred:
            with torch.no_grad():
                #self.actor.eval()
                mu, std = self.actor(x)
            #print("Mu, Log_Std: {} {}".format(mu, std))
            #self.get_actor_mean()
        else:
            self.actor.train()
            mu, std = self.actor(x)
        
        #Treat initial output std as log_std - prevent <= 0 std
        std = std.exp()
        act_dist = Normal(mu, std)
        
        u = act_dist.rsample()
        action = F.tanh(u)*self.action_scale+self.action_bias
        log_prob = act_dist.log_prob(u)
        jacobian = torch.log((1-torch.square(action))+eps)
        #if internalCall:
        jacobian = jacobian.sum(1, keepdim=True)
        log_prob -= jacobian
        
        if test:
            action = F.tanh(mu)*self.action_scale+self.action_bias
            self.actor.train()
        #  Internalcall used in training, evaluation and data collection has default False
        log_prob = log_prob.sum(1, keepdim=True)
        if internalCall:
            return action, log_prob
        else:
            #print("Action: {} State: {}".format(action, x))
            #self.get_actor_mean()
            return np.squeeze(action.cpu().numpy()), np.squeeze(log_prob.cpu().numpy())


    def _update_target(self):
        for q1, q1t in zip(self.qcritic.parameters(), self.tcritic.parameters()):
            q1t.data *= (1-self.tau)
            q1t.data += (self.tau)*q1.data
        for q2, q2t in zip(self.qcritic2.parameters(), self.tcritic2.parameters()):
            q2t.data *= (1-self.tau)
            q2t.data += self.tau*q2.data


    def train_step(self, replay_buffer, batch_size):
        state_set, action_set, reward_set, nstate_set, logprob_set, done_set = replay_buffer.sample(self.batch_size)
        state_set = torch.from_numpy(state_set).float().to(device)
        action_set = torch.from_numpy(action_set).float().to(device)
        reward_set = torch.from_numpy(reward_set).float().to(device)
        nstate_set = torch.from_numpy(nstate_set).float().to(device)
        done_set = torch.from_numpy(done_set).float().to(device)
        logprob_set = torch.from_numpy(logprob_set).float().to(device)
       
        alpha = self.log_alpha.exp()
        act, log_prob = self.predict(nstate_set, internalCall=True)
        qOut = torch.min(self.tcritic(nstate_set, act).detach(), self.tcritic2(nstate_set, act).detach())
        qOut -= alpha.detach()*(log_prob)
        reward_set = torch.unsqueeze(reward_set, 1)
        done_set = torch.unsqueeze(done_set, 1)
        target = reward_set+(1-done_set)*self.gamma*qOut
        q1loss = F.mse_loss(target, self.qcritic(state_set, action_set))
        q2loss = F.mse_loss(target, self.qcritic2(state_set, action_set))
        
        self.qc_opt.zero_grad()
        q1loss.backward()
        self.qc_opt.step()
        self.qc_opt2.zero_grad()
        q2loss.backward()
        self.qc_opt2.step()

        self.act_opt.zero_grad()
        act, log_prob = self.predict(state_set, False, True)
        min_q = torch.min(self.qcritic(state_set, act), self.qcritic2(state_set, act))
        actor_loss = (alpha.detach()*log_prob-min_q).mean() 
        actor_loss.backward()
        self.act_opt.step()
        
        self.alpha_opt.zero_grad()
        alphaLoss = (-self.log_alpha*(log_prob+self.target_ent).detach()).mean()
        alphaLoss.backward()
        self.alpha_opt.step()
        self._update_target()
        #print("Loss List: q1 {}, q2 {}, act {}, alpha {}".format(q1loss, q2loss, actor_loss, alphaLoss)) 
        #self.get_actor_mean()

    def save(self, path=None):
        if path is None:
            path = 'models/{}'.format(self.name)
            if os.path.isdir('models') is False:
                os.mkdir('models')
        torch.save({
            'actor': self.actor.state_dict(),
            'qcritic': self.qcritic.state_dict(),
            'qcritic2': self.qcritic2.state_dict(),
            'tcritic': self.tcritic.state_dict(),
            'tcritic2': self.tcritic.state_dict(),
            'log_alpha': self.log_alpha,
            'qopt': self.qc_opt.state_dict(),
            'qopt2': self.qc_opt2.state_dict(),
            'actOpt': self.act_opt.state_dict(),
            'alphaOpt': self.alpha_opt.state_dict()
            }, path) 
    

    def load(self, path=None):
        if path is None:
            path = 'models/{}'.format(self.name)
        load_dict = torch.load(path)
        self.actor.load_state_dict(load_dict['actor'])
        self.qcritic.load_state_dict(load_dict['qcritic'])
        self.qcritic2.load_state_dict(load_dict['qcritic2'])
        self.tcritic.load_state_dict(load_dict['tcritic'])
        self.tcritic2.load_state_dict(load_dict['tcritic2'])
        self.log_alpha = load_dict['log_alpha']
        self.qc_opt.load_state_dict(load_dict['qopt'])
        self.qc_opt2.load_state_dict(load_dict['qopt2'])
        self.act_opt.load_state_dict(load['actOpt'])
        self.alpha_opt.load_state_dict(load_dict['alphaOpt']) 
    
    def get_actor_mean(self):
        print("Start")
        for name, p in self.actor.named_parameters():
            print(name)
            print(p.data.mean())
            print("Gradient Data")
            print(p.grad.data.mean())
        print("End")
