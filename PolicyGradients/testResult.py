import pickle

obj = pickle.load(open('results/SAC_cheetah.p', 'rb'))
for entry in obj:
    print("Reward: {}, Step: {}".format(entry[1], entry[0]))
