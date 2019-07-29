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

#inductive learning
# if args.train_name is None:
#     if args.data_name == 'dream':
#         net = np.load(os.path.join(args.file_dir, 'data/dream/ind.dream3.csc'))
#         group = np.load(os.path.join(args.file_dir, 'data/dream/ind.dream3.allx'))
#         attributes =group.toarray().astype('float32')       
#     else:
#         args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
#         data = sio.loadmat(args.data_dir)
#         net = data['net']
#         if 'group' in data:
#             # load node attributes (here a.k.a. node classes)
#             attributes = data['group'].toarray().astype('float32')
#         else:
#             attributes = None
#         # check whether net is symmetric (for small nets only)
#     if False:
#         net_ = net.toarray()
#         assert(np.allclose(net_, net_.T, atol=1e-8))
#     #Sample train and test links
#     train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
# else:
#     args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
#     args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
#     train_idx = np.loadtxt(args.train_dir, dtype=int)
#     test_idx = np.loadtxt(args.test_dir, dtype=int)
#     max_idx = max(np.max(train_idx), np.max(test_idx))
#     net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
#     net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
#     net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
#     #Sample negative train and test links
#     train_pos = (train_idx[:, 0], train_idx[:, 1])
#     test_pos = (test_idx[:, 0], test_idx[:, 1])
#     train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)

# dream1: top 195
# dream3: top 334
# dream4: top 333 are TF

# Transductive learning
# For 1vs 1
if args.traindata_name is not None:
    trainNet = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name)))
    trainGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name)))
    allx =trainGroup.toarray().astype('float32')
    #deal with the features:
    trainAttributes = genenet_attribute(allx,334)   

    testNet = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.testdata_name)))
    testGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.testdata_name)))
    allxt =testGroup.toarray().astype('float32')
    #deal with the features:
    testAttributes = genenet_attribute(allxt,333)

    train_pos, train_neg, test_pos_un, test_neg_un = sample_neg(trainNet, 0.0, max_train_num=args.max_train_num)
    train_pos_un, train_neg_un, test_pos, test_neg = sample_neg(testNet, 1.0, max_train_num=args.max_train_num)



'''Train and apply classifier'''
Atrain = trainNet.copy()  # the observed network
Atest = testNet.copy()  # the observed network
Atest[test_pos[0], test_pos[1]] = 0  # mask test links
Atest[test_pos[1], test_pos[0]] = 0  # mask test links

train_node_information = None
test_node_information = None
if args.use_embedding:
    train_embeddings = generate_node2vec_embeddings(Atrain, 128, True, train_neg) #?
    train_node_information = train_embeddings
    test_embeddings = generate_node2vec_embeddings(Atest, 128, True, test_neg) #?
    test_node_information = test_embeddings
if args.use_attribute and trainAttributes is not None: 
    if train_node_information is not None:
        train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        test_node_information = np.concatenate([test_node_information, testAttributes], axis=1)
    else:
        train_node_information = trainAttributes
        test_node_information = testAttributes

train_graphs, test_graphs, max_n_label = links2subgraphsTran(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information, test_node_information)


# For 2 vs 1
if args.traindata_name2 is not None:
    trainNet2 = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name2)))
    trainGroup2 = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name2)))
    allx2 =trainGroup2.toarray().astype('float32')

    #deal with the features:
    trainAttributes2 = genenet_attribute(allx2,195)
    train_pos2, train_neg2, test_pos_un, test_neg_un = sample_neg(trainNet2, 0.0, max_train_num=args.max_train_num)

    Atrain2 = trainNet2.copy()  # the observed network

    train_node_information2 = None
    if args.use_embedding:
        train_embeddings2 = generate_node2vec_embeddings(Atrain2, 128, True, train_neg2) #?
        train_node_information2 = train_embeddings2
    if args.use_attribute and trainAttributes2 is not None: 
        if train_node_information2 is not None:
            train_node_information2 = np.concatenate([train_node_information2, trainAttributes2], axis=1)
        else:
            train_node_information2 = trainAttributes2

    train_graphs2, test_graphs, max_n_label = links2subgraphsTran(Atrain2, Atest, train_pos2, train_neg2, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information2, test_node_information)
    train_graphs = train_graphs + train_graphs2
    if train_node_information is not None:
        train_node_information = np.concatenate([train_node_information, train_node_information2], axis=0)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

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

