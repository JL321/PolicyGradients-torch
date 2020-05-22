import gym
from DDPG import DDPG
from SAC import SAC
from utils import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, default=True, help="Set train mode or evaluate")
args = parser.parse_args()

batch_size = 64

def clog_prob(val, mu=0, std=1):
    return np.sum(-np.log(np.sqrt(2*np.pi))-np.log(std**2)-(1/2*std)*np.square(val-mu))

def process_state(state):
    
    #state /= np.sqrt(np.sum(np.square(state)))
    return state

def fillBuffer(env, memory, action_space, length=1000, returnlp=True):
    
    state = process_state(env.reset())
    for _ in range(length):
        action = np.random.normal(size=action_space)
        if returnlp:
            log_prob = clog_prob(action)
        new_state, reward, done, _ = env.step(action)
        new_state = process_state(new_state)
        if returnlp:
            memory.add((state, action, reward, new_state, log_prob, done))
        else:
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
        action = model.predict(np.expand_dims(state, axis=0))
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

def runEp(env, memory, model, returnlp=False, returnReward=False):
     
    step = 0
    if returnReward:
        epReward = 0
    state = process_state(env.reset())
    for _ in range(5000): # Note that 5000 was an arbitrary choice
        if not returnlp:
            action = model.predict(np.expand_dims(state, axis=0))
        else:
            action, log_prob = model.predict(np.expand_dims(state, axis=0))
        action = np.clip(action, -5, 5)
        new_state, reward, done, _ = env.step(action)
        new_state = process_state(new_state)
        if not returnlp:
            memory.add((state, action, reward, new_state, done))
        else:
            memory.add((state, action, reward, new_state, log_prob, done))
        state = new_state
        step += 1 
        model.train_step(memory, batch_size) 
        if returnReward:
            epReward += reward
        if done:
            print("Done!")
            break
    if returnReward:
        return step, epReward, action
    return step

def train_model(env, memory, model, epIter=1000):
    reward_plot = [] # Format of (time step, ep_reward) pairs
    tStep = 0
    for i in range(epIter):
        if (i+1)%10 == 0: 
            step, epReward, act = runEp(env, memory, model, returnReward=True)
            reward_plot.append([tStep, epReward])
            if (i+1)%100 == 0:
                print("Last Act on Step {}: {}".format(act, i))
        else:
            step = runEp(env, memory, model, returnlp=True)
        if (i+1)%1000 == 0:
            print("Last action on last episode: {}".format(act))
            plt.plot([v[0] for v in reward_plot], [v[1] for v in reward_plot])
            plt.title("Episode Reward vs TimeStep")
            plt.show()
            model.save()
        tStep += step
    pickle.dump(reward_plot, open('results/SAC_cheetah.p', 'wb'))

def main():

    env = gym.make('HalfCheetah-v2')
    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0] 
    print("Action Space: {}".format(action_space))
    model = SAC(state_space, action_space, name='SAC_cheetah')
    if args.train:
        rmemory = ReplayBuffer(1e6)
        fillBuffer(env, rmemory, action_space)
        train_model(env, rmemory, model)
    else:
        model.load('models/SAC')
        recordEps(env, model, 'vid/halfCheetah.mp4')
        for _ in range(5):
            baseEp(env)

if __name__ == '__main__':
    main()

