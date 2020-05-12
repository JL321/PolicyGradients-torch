import pickle

obj = pickle.load(open('rewardDataWalker2d-v2-gpu.p', 'rb'))
for entry in obj:
    print("Reward: {}, Step: {}".format(entry[1], entry[0]))
