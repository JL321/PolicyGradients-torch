import gym
from DDPG import DDPG
from utils import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, default=True, help="Set train mode or evaluate")
args = parser.parse_args()

batch_size = 64
train = False

def process_state(state):
    
    #state /= np.sqrt(np.sum(np.square(state)))
    return state

def fillBuffer(env, memory, action_space, length = 1000):
    
    state = process_state(env.reset())
    for _ in range(length):
        action = np.random.normal(size=action_space)
        new_state, reward, done, _ = env.step(action)
        new_state = process_state(new_state)
        memory.add((state, action, reward, new_state, done))
        state = new_state
        if done:
            state = process_state(env.reset())

def recordEps(env, model, save_path=None):

    rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, save_path)
    state = env.reset()
    r = 0
    step = 0
    while(True):
        action = model.predict(np.expand_dims(state, axis=0), False)
        #rec.capture_frame()
        state, reward, done, _ = env.step(action)
        r += reward
        step += 1
        if done:
            rec.close()
            print("Reward at termination: {}".format(r))
            print("Avg Reward: {}".format(r/step))
            return

def baseEp(env):

    state = env.reset()
    r = 0
    while(True):
        action = np.random.normal(size = env.action_space.shape[0])
        state, reward, done, _ = env.step(action)
        r += reward
        if done:
            print("Random: Reward at Termination: {}".format(r))
            return

def runEp(env, memory, model, returnReward=False):
     
    step = 0
    if returnReward:
        epReward = 0
    state = process_state(env.reset())
    for _ in range(5000): # Note that 5000 was an arbitrary choice
        action = model.predict(np.expand_dims(state, axis = 0))
        action = np.clip(action, -5, 5)
        new_state, reward, done, _ = env.step(action)
        new_state = process_state(new_state)
        memory.add((state, action, reward, new_state, done))
        state = new_state
        step += 1 
        model.train_step(memory, batch_size) 
        if returnReward:
            epReward += reward
        if done:
            break
    if returnReward:
        return step, epReward, action
    return step

def train_model(env, memory, model, epIter = 1000):
    reward_plot = [] # Format of (time step, ep_reward) pairs
    tStep = 0
    for i in range(epIter):
        if (i+1)%10 == 0: 
            step, epReward, act = runEp(env, memory, model, True)
            reward_plot.append([tStep, epReward])
            if (i+1)%100 == 0:
                print("Last Act on Step {}: {}".format(act, i))
        else:
            step = runEp(env, memory, model)
        if (i+1)%1000 == 0:
            print("Last action on last episode: {}".format(act))
            plt.plot([v[0] for v in reward_plot], [v[1] for v in reward_plot])
            plt.title("Episode Reward vs TimeStep")
            plt.show('models/DDPG_pend')
            model.save()
        tStep += step
    pickle.dump(reward_plot, open('results/pendulum.p', 'wb'))

def main():

    env = gym.make('Pendulum-v0')
    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0] 
    model = DDPG(state_space, action_space)
    if train:
        rmemory = ReplayBuffer(1e6)
        fillBuffer(env, rmemory, action_space)
        train_model(env, rmemory, model)
    else:
        model.load('models/DDPG')
        recordEps(env, model, 'vid/pendulum.mp4')
        for _ in range(5):
            baseEp(env)

if __name__ == '__main__':
    main()

