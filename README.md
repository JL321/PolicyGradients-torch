# PolicyGradients-torch

Dependencies:
- gym
- numpy
- pytorch (scripts written in cuda 9.2)
- python3

This repo contains implementations of SAC and DDPG in pytorch. 

Structure:
- train.py is the main file to be run: use --test to load a model, and run the model for evaluation
- Hyperparameters are set by default, and are located in the train.py model
- util.py contains the experience buffer
- SAC.py and DDPG.py are self contained classes that include all content pertained to both algorithms
- SAC and DDPG can be interchangably loaded by changing the model imported in train.py

Additional guidelines to be added 
