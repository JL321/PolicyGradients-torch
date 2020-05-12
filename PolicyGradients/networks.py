import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):

    def __init__(self, state_space, action_space):
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        print("Verify State Space: {}".format(state_space))
        self.fc2 = nn.Linear(128, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, action_space)
        nn.init.uniform_(self.fc4.weight, -3*1e-3, 3*1e-3) 
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(300)
        self.b3 = nn.BatchNorm1d(400)

    def forward(self, x):

        x = self.b1(F.relu(self.fc1(x)))
        x = self.b2(F.relu(self.fc2(x)))
        x = self.b3(F.relu(self.fc3(x)))
        return F.tanh(self.fc4(x))

class Critic(nn.Module):

    def __init__(self, state_space, action_space):
        
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fcA1 = nn.Linear(action_space, 256)
        self.fcS1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, 1)
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(256)

    def forward(self, state, action):
        
        x = self.b1(F.relu(self.fc1(state)))
        aOut = self.fcA1(F.relu(action))
        sOut = self.b2(F.relu(self.fcS1(x)))
        comb = F.relu(aOut+sOut)
        out = F.relu(self.fc2(comb))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class DDPG():

    def __init__(self, state_space, action_space):
       
        self.exploreDistr = torch.distributions.normal.Normal(\
        torch.zeros(action_space), torch.ones(action_space))
        self.actor = Actor(state_space, action_space).to(device)
        self.critic = Critic(state_space, action_space).to(device)
        self.targActor = Actor(state_space, action_space).to(device)
        self.targCritic = Critic(state_space, action_space).to(device)
        self.actorOpt = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=0.005)
        self.criticOpt = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=0.005)
        self.loss = nn.MSELoss()
        self.gamma = 0.99
        self.rho = 0.001

    def predict(self, state, exploreNoise = True):
        
        self.actor.eval()
        state = torch.from_numpy(state).float().to(device)
        pred = self.actor(state).detach()
        if exploreNoise:
            pred += 0.1*self.exploreDistr.sample().to(device)
        return pred.numpy()[0]

    def train_step(self, replay, batch_count):
        
        self.actor.train()
        self.criticOpt.zero_grad()
        batch_sample = replay.sample(batch_count)
        state, action, reward, new_state, done = batch_sample
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        new_state = torch.from_numpy(new_state).float().to(device)
        done = torch.from_numpy(done).float().to(device)
        target = torch.unsqueeze(reward, 1)+torch.unsqueeze((1-done), 1)*self.gamma*(self.targCritic(new_state, self.targActor(new_state)))
        critic_loss = self.loss(target, self.critic(state, action))
        critic_loss.backward()
        self.criticOpt.step()

        self.actorOpt.zero_grad()
        policyLoss = -self.critic(state, self.actor(state)).mean()
        policyLoss.backward()
        self.actorOpt.step()
        self._update_target()

    def _update_target(self):
        for param1, param2 in zip(self.targActor.parameters(), self.actor.parameters()):
            param1.data *= (1-self.rho)
            param1.data += self.rho*param2.data
        for param1, param2 in zip(self.targCritic.parameters(), self.critic.parameters()):
            param1.data *= (1-self.rho)
            param1.data += self.rho*param2.data

    def save(self, path='models/DDPG'):
        
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')
        torch.save({'actor_weights': self.actor.state_dict(),
                    'critic_weights': self.critic.state_dict(),
                    'Coptimizer_param': self.criticOpt.state_dict(),
                    'Aoptimizer_param': self.actorOpt.state_dict()
                    }, path)
        print("Saved Model Weights!")

    def load(self, path='models/DDPG'):
        
        model_dict = torch.load(path)
        self.actor.load_state_dict(model_dict['actor_weights'])
        self.critic.load_state_dict(model_dict['critic_weights'])
        self.criticOpt.load_state_dict(model_dict['Coptimizer_param'])
        self.actorOpt.load_state_dict(model_dict['Aoptimizer_param'])
        print("Model Weights Loaded!")

class ReplayBuffer():

    def __init__(self, length = 10000):
    
        # Buffer Collection: (S, A, R, S', D)
        # Done represents a mask of either 0 and 1
        self.length = length
        self.buffer = []

    def add(self, sample):

        if (len(self.buffer) > self.length):
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size):
        
        idx = np.random.permutation(len(self.buffer))[:batch_size]
        state_b = []
        action_b = []
        reward_b = []
        nextstate_b = []
        done_b = []
        for i in idx:
            s, a, r, sp, d = self.buffer[i]
            state_b.append(s)
            action_b.append(a)
            reward_b.append(r)
            nextstate_b.append(sp)
            done_b.append(d)
        state_b = np.array(state_b)
        action_b = np.array(action_b)
        reward_b = np.array(reward_b)
        nextstate_b = np.array(nextstate_b)
        done_b = np.array(done_b)
        return (state_b, action_b, reward_b, nextstate_b, done_b)

