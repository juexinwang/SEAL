import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
#import cPickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--traindata-name', default='dream3', help='train network name')
parser.add_argument('--traindata-name2', default='dream1', help='train network name2')
parser.add_argument('--testdata-name', default='dream4', help='test network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

train_graphs=np.load('train_graphs.npy').tolist()
test_graphs=np.load('test_graphs.npy').tolist()
train_node_information=np.load('train_node_information.npy')
max_n_label = 11

# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 50
cmd_args.learning_rate = 1e-4
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if train_node_information is not None:
    cmd_args.attr_dim = train_node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))
best_loss = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    classifier.eval()
    test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')

