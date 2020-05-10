import gym
from networks import DDPG, ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64

def process_state(state):
    
    state /= np.sqrt(np.sum(np.square(state)))
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

def train_model(env, memory, model, epIter = 10000):
    reward_plot = [] # Format of (time step, ep_reward) pairs
    tStep = 0
    for i in range(epIter):
        if (i+1)%10 == 0: 
            step, epReward, act = runEp(env, memory, model, True)
            reward_plot.append([tStep, epReward])
            if (i+1)%100 == 0:
                print("Last Act on Step {}: {}".format(act, i))
        if (i+1)%1000 == 0:
            print("Last action on last episode: {}".format(act))
            plt.plot([v[0] for v in reward_plot], [v[1] for v in reward_plot])
            plt.title("Episode Reward vs TimeStep")
            plt.show()
            model.save()
        else:
            step = runEp(env, memory, model)
        tStep += step

def main():

    env = gym.make('BipedalWalker-v2')
    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0] 
    model = DDPG(state_space, action_space)
    rmemory = ReplayBuffer(1e6)
    fillBuffer(env, rmemory, action_space)
    train_model(env, rmemory, model)

if __name__ == '__main__':
    main()

