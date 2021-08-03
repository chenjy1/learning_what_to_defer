import os
import logging
import random
import numpy as np
import networkx as nx
import argparse

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import dgl
from ppo.framework import ProxPolicyOptimFramework
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage import RolloutStorage

from data.graph_dataset import get_er_15_20_dataset
from data.util import write_nx_to_metis

from env import MaximumIndependentSetEnv

import profile
import json

from parser import Parser
from Logg import Logger
from dqn import DQN
torch.set_printoptions(profile="full")

def main(args):
    
    if not (args.json is None):
        f = open(args.json)
        args = json.load(f)
        f.close()







    agent = DQN(args)
    # if agent.test is None:
    #     agent.test()
    # else:
    agent.learn()
    agent.test()
    # agent.save()
    # agent.plt()






if __name__ == '__main__':
    args = Parser(description='GIN').args
    
    main(args)