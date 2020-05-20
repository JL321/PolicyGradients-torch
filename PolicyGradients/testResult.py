import pickle

obj = pickle.load(open('results/pendulum.p', 'rb'))
for entry in obj:
    print("Reward: {}, Step: {}".format(entry[1], entry[0]))
