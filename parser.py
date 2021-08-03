"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()


    def _parse(self):

        self.parser.add_argument(
            '--json', type=str, default= None,
            help='load the json profile of args')

        self.parser.add_argument(
            '--test', type=str, default= None,
            help='given model address, skip training and test the model')

        # # dataset
        # self.parser.add_argument(
        #     '--dataset', type=str, default="n_20_method_ER_num_3000_p_1",
        #     help='name of dataset (default: n_30_method_ER_num_20)')
        # self.parser.add_argument(
        #     '--batch_size', type=int, default=128,
        #     help='batch size for training and validation (default: 32)')
        # self.parser.add_argument(
        #     '--fold_idx', type=int, default=0,
        #     help='the index(<10) of fold in 10-fold validation.')
        # self.parser.add_argument(
        #     '--filename', type=str, default="",
        #     help='output file')

        # # device
        # self.parser.add_argument(
        #     '--disable-cuda', action='store_true',
        #     help='Disable CUDA')
        # self.parser.add_argument(
        #     '--device', type=int, default=0,
        #     help='which gpu device to use (default: 0)')

        # # net
        # self.parser.add_argument(
        #     '--num_layers', type=int, default=4,
        #     help='number of layers (default: 10)')
        # self.parser.add_argument(
        #     '--num_mlp_layers', type=int, default=2,
        #     help='number of MLP layers(default: 2). 1 means linear model.')
        # self.parser.add_argument(
        #     '--hidden_dim', type=int, default=128,
        #     help='number of hidden units (default: 128)')

        # # # graph
        # # self.parser.add_argument(
        # #     '--graph_pooling_type', type=str,
        # #     default="sum", choices=["sum", "mean", "max"],
        # #     help='type of graph pooling: sum, mean or max')


        # # self.parser.add_argument(
        # #     '--neighbor_pooling_type', type=str,
        # #     default="sum", choices=["sum", "mean", "max"],
        # #     help='type of neighboring pooling: sum, mean or max')

        # # self.parser.add_argument(
        # #     '--neighbor_pooling_type_b', type=str,
        # #     default="sum", choices=["sum", "mean", "max"],
        # #     help='type of neighboring pooling: sum, mean or max')

        # # self.parser.add_argument(
        # #     '--neighbor_pooling_type_c', type=str,
        # #     default="sum", choices=["sum", "mean", "max"],
        # #     help='type of neighboring pooling: sum, mean or max')

        # # self.parser.add_argument(
        # #     '--learn_eps', action="store_true",
        # #     help='learn the epsilon weighting')

        # # learning
        # self.parser.add_argument(
        #     '--seed', type=int, default=0,
        #     help='random seed (default: 0)')
        # self.parser.add_argument(
        #     '--epochs', type=int, default=200,
        #     help='number of epochs to train (default: 350)')
        # self.parser.add_argument(
        #     '--lr', type=float, default= 0.005,
        #     help='learning rate (default: 0.01)')
        # self.parser.add_argument(
        #     '--final_dropout', type=float, default=0.0,
        #     help='final layer dropout (default: 0.5)')

        # done
        self.args = self.parser.parse_args()
