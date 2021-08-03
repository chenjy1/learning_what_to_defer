

from dataset import DataSet
import os
import logging
import random
import numpy as np
import networkx as nx
import argparse

from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from ppo.framework import ProxPolicyOptimFramework
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage import RolloutStorage

from data.graph_dataset import get_er_15_20_dataset
from data.util import write_nx_to_metis

from env import MaximumIndependentSetEnv, MaxCutEnv
from ultis import graphPartition, opt_sol, max_cut_solve, mvc_optimal_solve
import profile
import json

from parser import Parser
from Logg import Logger
import sys
import cplex
import pickle

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def hprint(info):
    print(f"{WARNING}{info}{ENDC}")



class DQN:
    # ok
    def __init__(self, args):

        self.train_sol_his = []
        self.val_sol_his = []
        self.args = args
        self.decode_args()
        self.loadDataset()
        self.loadModel()
        
        self.learn_step_counter = 0
        self.save_counter = 0
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.initial_learning_rate, 
        #                     eps = self.adam_epsilon,
        #                     weight_decay = self.weight_decay)
        self.loss_func = nn.MSELoss()
        self.best_score = 0
        self.show = 0

        if self.problem_type == "MIS":
            self.env = MaximumIndependentSetEnv(
                    max_epi_t = self.max_epi_t,
                    max_num_nodes = self.max_num_nodes,
                    hamming_reward_coef = self.hamming_reward_coef,
                    device = self.device
                    )
        elif self.problem_type == "MAXCUT":
            self.env = MaxCutEnv(
                    max_epi_t = self.max_epi_t,
                    max_num_nodes = self.max_num_nodes,
                    hamming_reward_coef = self.hamming_reward_coef,
                    device = self.device
                    )
        else:
            raise NotImplementedError


        self.rollout = RolloutStorage(
                max_t = self.max_rollout_t, 
                batch_size = self.rollout_batch_size, 
                num_samples = self.train_num_samples 
                )

        self.actor_critic = ActorCritic(
                actor_class = PolicyGraphConvNet,
                critic_class = ValueGraphConvNet, 
                max_num_nodes = self.max_num_nodes, 
                hidden_dim = self.hidden_dim,
                num_layers = self.num_layers,
                device = self.device
                )

        self.framework = ProxPolicyOptimFramework(
                actor_critic = self.actor_critic,
                init_lr = self.init_lr,
                clip_value = self.clip_value, 
                optim_num_samples = self.optim_num_samples,
                optim_batch_size = self.optim_batch_size,
                critic_loss_coef = self.critic_loss_coef, 
                reg_coef = self.reg_coef, 
                max_grad_norm = self.max_grad_norm, 
                device = self.device
                )    


    # ok
    def decode_args(self):
        self.device = self.args["device"]
        self.hamming_reward_coef = self.args["hamming_reward_coef"]
        self.num_layers = self.args["num_layers"]
        self.input_dim = self.args["input_dim"]
        self.output_dim = self.args["output_dim"]
        self.hidden_dim = self.args["hidden_dim"]
        self.init_lr = self.args["init_lr"]
        self.max_epi_t = self.args["max_epi_t"]
        self.max_rollout_t = self.args["max_rollout_t"]
        self.max_update_t = self.args["max_update_t"]
        self.gamma = self.args["gamma"]
        self.clip_value = self.args["clip_value"]
        self.optim_num_samples = self.args["optim_num_samples"]
        self.critic_loss_coef = self.args["critic_loss_coef"]
        self.reg_coef = self.args["reg_coef"]
        self.max_grad_norm = self.args["max_grad_norm"]
        self.vali_freq = self.args["vali_freq"]
        self.log_freq = self.args["log_freq"]
        self.dataset = self.args["dataset"]
        self.graph_type = self.args["graph_type"]
        self.min_num_nodes = self.args["min_num_nodes"]
        self.max_num_nodes = self.args["max_num_nodes"]
        self.rollout_batch_size = self.args["rollout_batch_size"]
        self.eval_batch_size = self.args["eval_batch_size"]
        self.optim_batch_size = self.args["optim_batch_size"]
        self.init_anneal_ratio = self.args["init_anneal_ratio"]
        self.max_anneal_t = self.args["max_anneal_t"]
        self.anneal_base = self.args["anneal_base"]
        self.train_num_samples = self.args["train_num_samples"]
        self.eval_num_samples = self.args["eval_num_samples"]
        self.best_vali_sol = self.args["best_vali_sol"]
        self.num_eval_graphs = self.args["num_eval_graphs"]

        self.problem_type = self.args["problem_type"]  
        self.model_name = self.args["model_name"]   
        self.version = self.args["version"]        
        self.model = self.args["model"]
        self.adam_epsilon = self.args["adam_epsilon"]
        self.weight_decay = self.args["weight_decay"]
        self.train_save_path = self.args["train_save_path"]
        self.val_save_path = self.args["val_save_path"]
        self.test_save_path = self.args["test_save_path"]
        self.seed = self.args["seed"]
        self.with_sep = self.args["with_sep"]

        sys.stdout = Logger("name_{}_v_{}_".format(self.model_name, self.version))


        print('show all arguments configuration...')
        print(self.args)
        for i in self.args:
            print("{} : {}".format(i, self.args[i]))

        print()
        print("configuration end")


    def loadDataset(self):


        # all graph in the same dataset should have the same number of nodes


        if 1 :
            self.trainDataset = DataSet(seed = self.seed, device = self.device, name = self.train_save_path, with_sep = self.with_sep)
            self.trainLoader = self.trainDataset.getloader(self.rollout_batch_size)
            self.trainDataset.print_info()
            if os.path.isfile("{}.opt".format(self.train_save_path)):
                print(1)
                with open("{}.opt".format(self.train_save_path), 'rb+') as f:
                    self.train_opt = pickle.load(f)
            else:  
                # print(len(self.trainDataset))
                # print(self.trainDataset[0])
                # print(2)

                self.train_opt = opt_sol(self.trainDataset)
                with open("{}.opt".format(self.train_save_path), 'wb+') as f:
                    pickle.dump(self.train_opt, f)

            print(self.train_opt)
            print()
            # exit()
            self.valDataset = DataSet(seed = self.seed, device = self.device, name = self.val_save_path, with_sep = self.with_sep)
            self.valLoader = self.valDataset.getloader(self.eval_batch_size)
            self.valDataset.print_info()

            if os.path.isfile("{}.opt".format(self.val_save_path)):
                with open("{}.opt".format(self.val_save_path), 'rb+') as f:
                    self.val_opt = pickle.load(f)
            else:
                self.val_opt = opt_sol(self.valDataset)
                with open("{}.opt".format(self.val_save_path), 'wb+') as f:
                    pickle.dump(self.val_opt, f)

            print(self.val_opt)
            print()



        self.testDataset = DataSet(seed = self.seed, device = self.device, name = self.test_save_path, with_sep = self.with_sep)
        self.testLoader = self.testDataset.getloader(self.eval_batch_size)
        self.testDataset.print_info()

        if os.path.isfile("{}.opt".format(self.test_save_path)):
            with open("{}.opt".format(self.test_save_path), 'rb+') as f:
                self.test_opt = pickle.load(f)
        else:
            self.test_opt = opt_sol(self.testDataset)
            with open("{}.opt".format(self.test_save_path), 'wb+') as f:
                pickle.dump(self.test_opt, f)

        print(self.test_opt)
        print()

        # exit()

        # self.cplexsolve()

        # cnt = 0
        # total = 0
        # for i in self.trainDataset:
        #     cnt += max_cut_solve(i)[-1]
        #     total += i.num_nodes()
        # self.train_opt = (total - cnt)/len(self.trainDataset)

        # cnt = 0
        # total = 0
        # for i in self.valDataset:
        #     cnt += max_cut_solve(i)[-1]
        #     total += i.num_nodes()
        # self.val_opt = (total - cnt)/len(self.valDataset)


        # cnt = 0
        # total = 0
        # for i in self.trainDataset:
        #     cnt += sum(mvc_optimal_solve(i))
        #     total += i.num_nodes()
        # self.train_opt = (total - cnt)/len(self.trainDataset)

        # cnt = 0
        # total = 0
        # for i in self.valDataset:
        #     cnt += sum(mvc_optimal_solve(i))
        #     total += i.num_nodes()
        # self.val_opt = (total - cnt)/len(self.valDataset)

        # cnt = 0
        # total = 0
        # for i in self.testDataset:
        #     cnt += sum(mvc_optimal_solve(i))
        #     total += i.num_nodes()
        # self.test_opt = (total - cnt)/len(self.testDataset)

        # hprint("train_opt = {}, val_opt = {}".format(self.train_opt, self.val_opt))


        # hprint("train_opt = {}, val_opt = {}, test_opt = {}".format(self.train_opt, self.val_opt, self.test_opt))



        # exit()

        # self.generate_envs()


    # ok
    def loadModel(self):

        if self.model == -1:
            pass


        else:
            raise NotImplementedError()


    def learn(self):


        for update_t in range(self.max_update_t):
            if update_t == 0 or torch.all(done).item():
                try:
                    g, sep_g = next(train_data_iter)
                except:
                    train_data_iter = iter(self.trainLoader)
                    g, sep_g = next(train_data_iter)
                # g = dgl.add_self_loop(g)
                g.set_n_initializer(dgl.init.zero_initializer)
                ob = self.env.register(g, num_samples = self.train_num_samples)
                self.rollout.insert_ob_and_g(ob, g)


                # g.to(self.device)

            for step_t in range(self.max_rollout_t):
                # get action and value prediction
                with torch.no_grad():
                    (action, 
                    action_log_prob, 
                    value_pred, 
                    ) = self.actor_critic.act_and_crit(ob, g)

                # step environments
                ob, reward, done, info = self.env.step(action)

                # insert to rollout
                self.rollout.insert_tensors(
                    ob, 
                    action,
                    action_log_prob, 
                    value_pred, 
                    reward, 
                    done
                    )

                if torch.all(done).item():
                    avg_sol = info['sol'].max(dim = 1)[0].mean().cpu()
                    break

            # compute gamma-decayed returns and corresponding advantages
            self.rollout.compute_rets_and_advantages(self.gamma)

            # update actor critic model with ppo
            actor_loss, critic_loss, entropy_loss = self.framework.update(self.rollout)

            if (update_t + 1) % self.log_freq == 0:
                self.print_log(update_t, avg_sol, actor_loss, critic_loss, entropy_loss)

    def eval(self, mode):

        if mode == "val":
            dataset = self.valDataset
        elif mode == "test":
            dataset = self.testDataset
        elif mode == "train":
            dataset = self.trainDataset
        else:
            raise NotImplementedError

        self.actor_critic.eval()
        cum_cnt = 0
        cum_eval_sol = 0.0
        for g, sep_g in dataset:
            g.set_n_initializer(dgl.init.zero_initializer)
            ob = self.env.register(g, num_samples = self.eval_num_samples)
            while True:
                with torch.no_grad():
                    action = self.actor_critic.act(ob, g)

                ob, reward, done, info = self.env.step(action)
                if torch.all(done).item():
                    cum_eval_sol += info['sol'].max(dim = 1)[0].sum().cpu()
                    cum_cnt += g.batch_size
                    break
        
        self.actor_critic.train()
        avg_eval_sol = cum_eval_sol / cum_cnt
        return avg_eval_sol


    def test(self):
        sol = self.eval("test")
        hprint("test stats...")
        hprint("sol: {:.4f}".format(sol.item()))

    def print_log(self, update_t, avg_sol, actor_loss, critic_loss, entropy_loss):
        print("update_t: {:05d}".format(update_t + 1))
        print("train stats...")
        print(
            "sol: {:.4f}, "
            "actor_loss: {:.4f}, "
            "critic_loss: {:.4f}, "
            "entropy: {:.4f}".format(
                avg_sol,
                actor_loss.item(),
                critic_loss.item(),
                entropy_loss.item()
                )
            )
        if (update_t + 1) % self.vali_freq == 0:
            sol = self.eval("val")
            self.val_sol_his.append(sol/2)
            train_sol = self.eval("train")
            self.train_sol_his.append(train_sol/2)
            self.plt_and_save()
            hprint("vali stats...")
            hprint("sol: {:.4f}".format(sol.item()))
            self.save(sol.item())


    def save(self, score, comment = None):



        if comment is None:
            comment = self.save_counter
            self.save_counter += 1

        # ovvo = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        
        save_path = "model/name_{}_v_{}_score_{}_actor_net_comment_{}".format(self.model_name, self.version, score, comment)

        torch.save(self.actor_critic.actor_net.state_dict(), save_path)

        save_path = "model/name_{}_v_{}_score_{}_critic_net_comment_{}".format(self.model_name, self.version, score, comment)

        torch.save(self.actor_critic.critic_net.state_dict(), save_path)


    def cplexsolve(self):
        # "/root/MVC/hmc/data/eco_data/_graphs/validation/BA_20spin_m4_100graphs.pkl_dgl_graph.bin",
        # maxcut : 46.69

        # "/root/MVC/hmc/data/eco_data/_graphs/testing/BA_40spin_m4_50graphs.pkl_dgl_graph.bin",
        # maxcut : 105.66

        cnt = 0
        total = 0
        for i in self.trainDataset:
            cnt += max_cut_solve(i)[-1]
            total += i.num_nodes()
        self.train_opt = cnt/len(self.trainDataset)

        hprint("train_opt = {}".format(self.train_opt))

        cnt = 0
        total = 0
        for idx, i in enumerate(self.valDataset):
            print("idx = ", idx)
            cnt += max_cut_solve(i)[-1]
            total += i.num_nodes()
        self.val_opt = cnt/len(self.valDataset)
        hprint("val_opt = {}".format(self.val_opt))


    def plt_and_save(self, comment = None):
        
        x = np.linspace(0, len(self.train_sol_his), len(self.train_sol_his))

        plt.plot(x, 0*x + sum(self.train_opt["cplex_sol"])/len(self.train_opt["cplex_sol"]), c = 'red', linestyle = '--', label = 'train-cplex_sol')
        plt.plot(x, 0*x + sum(self.train_opt["greedy_sol"])/len(self.train_opt["cplex_sol"]), c = 'red', linestyle = ':', label = 'train-greedy_sol')
        plt.plot(self.train_sol_his, c ='red', label = 'train')

        plt.plot(x, 0*x + sum(self.val_opt["cplex_sol"])/len(self.val_opt["cplex_sol"]), c = 'blue', linestyle = '--', label = 'train-cplex_sol')
        plt.plot(x, 0*x + sum(self.val_opt["greedy_sol"])/len(self.val_opt["cplex_sol"]), c = 'blue', linestyle = ':', label = 'train-greedy_sol')
        plt.plot(self.val_sol_his, c ='blue', label = 'val')
        plt.ylabel('Value {}'.format(self.problem_type))
        plt.xlabel('Times (freq = {})'.format(self.vali_freq))
        plt.savefig("img/val_{}_v_{}_{}.png".format(self.model_name, self.version, self.problem_type))
        plt.close()