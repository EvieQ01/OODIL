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
from sklearn.cluster import KMeans# import ant
import sys
sys.path.append("../")
sys.path.append("../envs")
import pdenv.gym_panda
from sklearn import metrics
# sys.path.append('../all_envs')
# import swimmer
# import walker
from visual_utils import visualize_obs
from SimCLR_traj_cluster import *
from gail_airl_ppo.buffer import SerializedBuffer, ConcatStateBuffers

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="panda-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--save_path', type=str, default= 'temp', metavar='N',
                    help='path to save demonstrations on')
parser.add_argument('--xml', type=str, default= None, metavar='N',
                    help='For diffent dynamics')
parser.add_argument('--demo_files', nargs='+', default=['buffers/panda-v0/size100000_able_panda_init0.5.pth'])
parser.add_argument('--test_demo_files', nargs='+')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0])
parser.add_argument('--mode', default='simclr')
parser.add_argument('--temperature', type=float, default=0.01, help='the temperature for simclr loss') # 
parser.add_argument('--discount_train', action='store_true')
parser.add_argument('--out_type', default='c_n', help='the output as representation')
parser.add_argument('--dist_type', default='l2', choices=['l2', 'cos'])
parser.add_argument('--downsample_stride', type=int, default=20, help='the stride of downsample')
parser.add_argument('--rnn_layers', type=int, default=1, help='the layers of lstm')
parser.add_argument('--source-str', type=str, default=None, help='the source of cluster')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of print logs')

# Model parameters
parser.add_argument('--lamda', type=float, default=1,
                    help='coefficient of the reconstruction loss')
parser.add_argument('--beta', type=float, default=.3,
                    help=('coefficient of the regularization term on '
                            'clustering'))
parser.add_argument('--H_in', default=0,
                    help='input dim, adjusted to demonstrations')
# donot needed if there is only one layer
# parser.add_argument('--hidden-dims', default=128 ,# origin [500, 500, 2000],
#                     help='hidden dim')
parser.add_argument('--latent_dim', type=int, default=128, # origin:10
                    help='latent space dimension')
parser.add_argument('--n-clusters', type=int, default=8,
                    help='number of clusters in the latent space')

# training parameters
parser.add_argument('--max_iteration', type=int, default=2000, help='the number of iterations')
parser.add_argument('--simclr_warmup', type=int, default=200, help='the number of iterations')
parser.add_argument('--load_only', action='store_true', help='whether to load pretrained model')

# saving parameters
parser.add_argument('--file_name', type=str, help='the file to save re-split model',default='from_4_6_0_split0.pth')
parser.add_argument('--dump', action='store_true')
args = parser.parse_args()


logger = CompleteLogger(f'logs/{args.env_name}/resplit-dataset-DCN/{args.mode}_{args.n_clusters}cluster/' \
            + "source-{}-temperature{}-beta-{}-batch{}-stride{}-ratio{}-save{}".format(args.source_str, args.temperature, args.beta, args.batch_size, args.downsample_stride, args.ratio, args.file_name))
save_rnn_path = logger.get_checkpoint_path(f'seed_{args.seed}_rnn_{args.out_type}_DCN_model')
print('sources: ', args.source_str)
# re-define datasets
if torch.cuda.is_available():
    DEVICE = 'cuda'
    # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"
print("current deveice: ", DEVICE)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
def load_demos(demo_files, ratio):
    idx = 0
    all_buffers = []
    single_domain_pairs = []
    for demo_file in demo_files:
        if 'buffers' not in demo_file:
            demo_file = os.path.join("buffers", demo_file)
        buffer_exp = SerializedBuffer(
            path=demo_file,
            device="cpu")
        all_buffers.append(buffer_exp)
        use_num = int(len(buffer_exp.states)*ratio[idx]) 
        traj_len = buffer_exp.traj_len[:int(len(buffer_exp.traj_len)*ratio[idx])]
        idx += 1
        all_pairs = buffer_exp.states[:use_num] # 10000 * 6

        #### reshape as (traj, step_per_traj, state_dim) ###########
        begin = 0
        for single_len in traj_len:
            single_domain_pairs.append(all_pairs[begin : begin + single_len])
            begin += single_len
        print("=>load from {}, {} trajs, {} state pairs in all".format(demo_file, len(traj_len), use_num))
        ## visualize
        visualize_obs(all_pairs,path=logger.get_image_path(f'loaded_demo.png' ))
    concat_buffer = ConcatStateBuffers(all_buffers)
    del all_buffers
    return single_domain_pairs, concat_buffer # (traj, step_per_traj, state_dim) (domains)


def save_split_buffers(split_clique, buffer_all: ConcatStateBuffers, path='buffers/resplit'):
    '''
    split_clique: list of traj id. ([1, 2, 3], [4, 5, 6], ...)
    '''
    cluster_id = 0
    for traj_id_list in split_clique:
        if len(traj_id_list) == 0:
            continue
        sub_buffer = buffer_all.get_sub_buffer(traj_idx_list=traj_id_list)
        sub_buffer.save(os.path.join(
                            'buffers',
                            args.env_name,
                            args.file_name.replace('split0', f'split{cluster_id}')
                        ), true_length=sum(sub_buffer.traj_len))
        print(f"=> save cluster {cluster_id}: traj x {len(sub_buffer.traj_len)}")

        cluster_id += 1

# load all demos
demos_all, concat_buffer = load_demos(args.demo_files, args.ratio)
test_demos = []
test_init_obs = []

args.H_in = demos_all[0].shape[-1]
json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
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
            # cluster_loss = torch.log(cluster_loss + 1)
        else:
            # part 1: do not compute cluster loss before warm up finish
            feat_1, _ = model(batch_1, training=False)# (B, latent_dim)
            feat_2, _ = model(batch_2, training=False)# (B, latent_dim)
            cluster_loss = 0.
        
        # part 2: compute simclr loss
        simclr_loss = loss_func(feat_1, feat_2, args.batch_size, temperature=args.temperature)
        optimizer.zero_grad()

        # sum 2 parts. 
        # simclr_loss = torch.log(simclr_loss + 1)
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
if args.dump:
    save_split_buffers(split_clique=split_clique, buffer_all=concat_buffer)
else:
    print("=> didnt save")

si_score = metrics.silhouette_score(features_all, cluster_id_all, metric='cosine')
ch_score = metrics.calinski_harabasz_score(features_all, cluster_id_all)
print(f'silhoueete score: {si_score}; calinki score: {ch_score}')
