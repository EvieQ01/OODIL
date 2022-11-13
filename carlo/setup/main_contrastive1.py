from tqdm import trange
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import random
import argparse

from pdb import set_trace

from logger import *
import json
import gym

import pdb
import torch
import numpy as np

import pickle
import copy
import sys
sys.path.append("../")
sys.path.append("../envs")
import envs
# sys.path.append('../all_envs')
# import swimmer
# import walker
from utils import visualize_obs
from SimCLR_traj_cluster import *
from sklearn import metrics
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--xml', type=str, default= None, metavar='N',
                    help='For diffent dynamics')
parser.add_argument('--demo_files', nargs='+')
parser.add_argument('--ratio', type=float, nargs='+')
parser.add_argument('--mode', default='simclr')
parser.add_argument('--temperature', type=float, default=0.1, help='the temperature for simclr loss') # 
parser.add_argument('--dump', action='store_true')
parser.add_argument('--out_type', default='c_n', help='the output as representation')
parser.add_argument('--dist_type', default='l2', choices=['l2', 'cos'])
parser.add_argument('--downsample_stride', type=int, default=20, help='the stride of downsample')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of print logs')

# Model parameters
parser.add_argument('--rnn_layers', type=int, default=1, help='the layers of lstm')
parser.add_argument('--beta', type=float, default=1.,
                    help=('coefficient of the regularization term on '
                            'clustering'))
parser.add_argument('--H_in', default=0,
                    help='input dim, adjusted to demonstrations')
# donot needed if there is only one layer
# parser.add_argument('--hidden-dims', default=128 ,# origin [500, 500, 2000],
#                     help='hidden dim')
parser.add_argument('--latent_dim', type=int, default=128, # origin:10
                    help='latent space dimension')
parser.add_argument('--n-clusters', type=int, default=10,
                    help='number of clusters in the latent space')

# training parameters
parser.add_argument('--max_iteration', type=int, default=2000, help='the number of iterations')
parser.add_argument('--simclr_warmup', type=int, default=200, help='the number of iterations')
parser.add_argument('--load_only', action='store_true', help='whether to load pretrained model')
args = parser.parse_args()


logger = CompleteLogger(f'log/resplit-dataset-DCN/{args.env_name}/{args.mode}_{args.n_clusters}cluster/' \
            + "temperature{}-beta-{}-batch{}-stride{}-ratio{}".format(args.temperature, args.beta, args.batch_size, args.downsample_stride, args.ratio))
save_rnn_path = logger.get_checkpoint_path(f'seed_{args.seed}_rnn_{args.out_type}_DCN_model')
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"
print("current deveice: ", DEVICE)

random.seed(args.seed)
torch.manual_seed(args.seed)
def load_demos(demo_files, ratio):
    all_demos = []
    idx = 0
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        # use fix ratio for every domain
        use_num = int(len(raw_demos['obs'])*ratio[idx])
        idx += 1
        demo_single = raw_demos['obs'][:use_num]
        
        for i in range(use_num):
            if i % 50 == 0:
                print("len demos {}:{}".format(i, len(demo_single[i]))) 
        print("=>load from {}, {} trajs".format(demo_file, use_num))
        all_demos = all_demos + demo_single
    return all_demos

def load_pairs(demo_files, ratio):
    all_pairs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        for i in range(int(len(raw_demos['obs'])*ratio)):
            obs = np.array(raw_demos['obs'][i])
            next_obs = np.array(raw_demos['next_obs'][i])
            all_pairs.append(np.reshape(np.concatenate([obs, next_obs], axis=1), (obs.shape[0], 2, -1)))
    return np.concatenate(all_pairs, axis=0)


def save_split_clique(split_clique, demos_all, demos_init_obs=[], type=None):
    raw_demos = {}
    for i in range(len(split_clique)):
        save_demo_path = '../demo/' + args.env_name + '/re_split_{}_{}_DCN_batch_00_temperature{}-beta-{}-batch{}-stride{}-ratio{}-k{}.pkl'.format(args.mode, i, args.temperature, args.beta, args.batch_size, args.downsample_stride, args.ratio, args.n_clusters)
        os.makedirs('../demo/' + args.env_name, exist_ok=True)
        if len(split_clique[i]) == 0: # empty clique
            continue
        raw_demos['obs'] = [demos_all[idx] for idx in split_clique[i]]
        os.makedirs(logger.get_image_path(f'split_for_{len(split_clique)}/' ), exist_ok=True)
        visualize_obs([demos_all[idx][:, 0:2] for idx in split_clique[i]], path=logger.get_image_path(f'split_for_{len(split_clique)}/split_{i}'))
        raw_demos['mapping'] = split_clique
        if args.dump:
            pickle.dump(raw_demos, open(save_demo_path, 'wb'))
            print("=>save cluster {}: {} traj at {}".format(i, len(raw_demos['obs']), save_demo_path))
        else:
            print("=>didn't save cluster {}: {} traj".format(i, len(raw_demos['obs'])))

# load all demos
demos_all = load_demos(args.demo_files, args.ratio)
test_demos = []

args.H_in = demos_all[0].shape[-1]
json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
# H_out = 128
# train
train_dataset = preDataset(demos_all, stride=args.downsample_stride)
model = DCN(args, device=DEVICE).to(DEVICE)
if args.load_only:   
    model = torch.load(save_rnn_path)
    args.max_iteration = 0

lossLR = loss_simclr().to(DEVICE)
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
# train self-supervised
def train(model, loss_func, optimizer, args):
    for iter in trange(args.max_iteration):
        # initialize after warmup
        if iter == args.simclr_warmup:
            
            features_all, cluster_id_all = featurize_with_dcn(demos_all=demos_all, model=model, mode=args.out_type, stride=args.downsample_stride)
            model.kmeans.init_cluster(features_all.numpy())
        batch_1, batch_2 = train_dataset.get_paired_batch(batch_size=args.batch_size) # L, B, H_in
        batch_1 = batch_1.to(DEVICE)
        batch_2 = batch_2.to(DEVICE)

        model.train()

        if iter > args.simclr_warmup:
            # part 1: compute cluster loss
            feat_1, cluster_loss_1 = model(batch_1)# (B, latent_dim)
            feat_2, cluster_loss_2 = model(batch_2)# (B, latent_dim)
            cluster_loss = cluster_loss_1 + cluster_loss_2
        else:
            # part 1: do not compute cluster loss before warm up finish
            feat_1, _ = model(batch_1, training=False)# (B, latent_dim)
            feat_2, _ = model(batch_2, training=False)# (B, latent_dim)
            cluster_loss = 0.
        
        # part 2: compute simclr loss
        simclr_loss = loss_func(feat_1, feat_2, args.batch_size, temperature=args.temperature)
        optimizer.zero_grad()

        # sum 2 parts. 
        loss = simclr_loss + cluster_loss
        loss.backward()
        optimizer.step()
        if iter > args.simclr_warmup and iter % args.print_freq == 0:
            print("iter: ", iter, "\tsimclr_loss: ", simclr_loss.detach().item(), \
                            "\tcluster_loss: ", cluster_loss.detach().item(),
                            "\toverall loss:", loss.detach().item())
        elif iter % args.print_freq == 0:
            print("Pretrained iter: ", iter, "\tsimclr_loss: ", simclr_loss.detach().item(), \
                            "\toverall loss:", loss.detach().item())
    return model

# Training
model = train(model=model, loss_func=lossLR, optimizer=optimizer, args=args)
torch.save(model, save_rnn_path)
print('=>save rnn model at: ', save_rnn_path)
# Save clusters
features_all, cluster_id_all = featurize_with_dcn(demos_all=demos_all, model=model, mode=args.out_type, stride=args.downsample_stride)
split_clique = []
for label in range(args.n_clusters):
    split_clique.append(np.argwhere(cluster_id_all == label).squeeze(-1))
    print(f'cluster:{label}, count {len(split_clique[-1])}: {split_clique[-1]}')
si_score = metrics.silhouette_score(features_all, cluster_id_all, metric='cosine')
ch_score = metrics.calinski_harabasz_score(features_all, cluster_id_all)
print(f'silhoueete score: {si_score}; calinki score: {ch_score}')

save_split_clique(split_clique=split_clique, demos_all=demos_all)
