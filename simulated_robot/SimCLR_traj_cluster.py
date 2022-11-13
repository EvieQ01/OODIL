import pdb
import torch
import numbers
import numpy as np
import torch.nn as nn
from kmeans import batch_KMeans
from lstm_encoder import *# unsupervised learning

class DCN(nn.Module):

    def __init__(self, args, device='cuda'):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = device

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))


        self.kmeans = batch_KMeans(args)
        self.autoencoder = lstm_encoder(input_size=args.H_in, num_layers=args.rnn_layers).to(self.device)

    def forward(self, X, mode='out', training=True):
        '''
        x: shape as (L, B, H)
        latent: shape as (B, latent_dim)
        mode == 'out', then return out
        mode == 'c_n', then return c_n
        '''
        batch_size = X.size()[1]
        o_n, (h_n, c_n) = self.autoencoder(X)
        latent_X = torch.flatten(o_n, start_dim=1) if mode == 'out' else torch.flatten(c_n, start_dim=1)
        latent_X = latent_X if self.args.dist_type == 'l2' else F.normalize(latent_X, dim=-1)

        # if training
        if training:
            
            # [Step-1] Update the assignment results
            cluster_id = self.kmeans.update_assign(latent_X.detach().cpu().numpy())

            # [Step-2] Update clusters in bath Kmeans
            elem_count = np.bincount(cluster_id, minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(latent_X.detach().cpu().numpy()[cluster_id == k], k)

            # [Step-3] Regularization term on clustering, return the loss
            dist_loss = torch.tensor(0.).to(self.device)
            clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
            for i in range(batch_size):
                diff_vec = latent_X[i] - clusters[cluster_id[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                                diff_vec.view(-1, 1))
                dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

            return latent_X, dist_loss
        else:
            # [Step-1] Update the assignment results
            cluster_id = self.kmeans.update_assign(latent_X.detach().cpu().numpy())
            return latent_X, cluster_id

def featurize_with_dcn(demos_all, model:nn.Module, mode='c_n', sample_count=5, device='cuda',stride=20):
    '''
    mode: whether to use out/c_n of RNN as the representation.
    return: all_features (n, latent_dim)
            all_ids      (n)
    '''
    model.eval()
    features = torch.empty((0, model.args.latent_dim)) # 0 * H_out
    len_of_traj = [len(tra) for tra in demos_all]
    steps = min(len_of_traj)
    cluster_ids = np.zeros(len(demos_all), dtype=int)
    idx = 0
    for demo in demos_all:
        test_traj_segment = torch.zeros((steps // stride, demo.shape[-1]))
        seg_generator = traj_segment_generator(demo, max_len=steps // stride, stride=stride)
        # get mean downsample
        for i in range(sample_count):
            single_downsample = seg_generator.get_downsample()
            try:
                assert test_traj_segment.shape[0] == single_downsample.shape[0]   
                test_traj_segment += single_downsample # len, obs_dim
            except:
                print('test_traj_segment', test_traj_segment, 'single_downsample', single_downsample)             
        test_traj_segment /= sample_count

        test_traj_segment = (test_traj_segment).to(device)
        # forward
        with torch.no_grad():
            feature, cluster_id = model(torch.unsqueeze(test_traj_segment, dim=1), mode=mode, training=False) # (step, 1, H_in)
            
            features = torch.cat((features, feature.cpu()), dim=0)
        cluster_ids[idx] = cluster_id
        idx += 1
    model.train()
    return features, cluster_ids #(n, latent_dim), (n)

class traj_segment_generator():
    '''
    traj: # (steps，state_dim)
    Generate downsample of one traj.
    '''
    def __init__(self, traj: np.ndarray, stride=20, max_len=None) -> None:
        self.traj = traj                
        self.total_len = len(traj)
        self.stride = stride
        self.max_len = max_len
        # self.max_possible_downsample = len(traj) // stride
        self.max_possible_downsample = stride

    def get_downsample(self, begin_index=None)-> np.ndarray: 
        '''
        return a 2 dimentional downsample
        '''
        if begin_index is not None:
            assert begin_index < self.max_possible_downsample
        else:
            begin_index = np.random.randint(self.stride)
        ind = np.arange(begin_index, self.total_len, self.stride)
        # if all trajs have different length, then constrain it to the same length
        if self.max_len is not None:
            drop_count = len(ind) - self.max_len
            if drop_count > 0:
                random_begin = np.random.randint(drop_count)
                ind = ind[random_begin : random_begin + self.max_len]

        return self.traj[ind]
    def get_downsample_all(self)-> np.ndarray: 
        '''
        return 3 dimentional: counts * 2 dimentional downsample (counts, steps，state_dim)
        '''
        downsample_all = np.expand_dims(self.get_downsample(begin_index=0), axis=0)
        for begin_index in range(1, self.max_possible_downsample):
            downsample_all = np.concatenate((downsample_all, np.expand_dims(self.get_downsample(begin_index=begin_index), axis=0)), axis=0)
        return downsample_all
            

class preDataset():
    '''
    A dataset that contains all trajectories and can genenrate 
        2 batches of tensor at a time.
    trajs: is (n, steps, state_dim)
    '''
    def __init__(self, trajs, stride=20) -> None:
        # (50 trajs, 20 subsamples, 50 steps, 18 dims)
        # different length
        len_of_traj = [len(tra) for tra in trajs]
        steps = min(len_of_traj)
        state_dim = trajs[0].shape[-1]
        self.all_traj_segments = np.empty((0, stride, steps // stride, state_dim))
        for traj in trajs:
            
            downsample_all = np.expand_dims(traj_segment_generator(traj=np.array(traj), stride=stride, max_len=steps // stride).get_downsample_all(), axis=0)
            self.all_traj_segments = np.concatenate((self.all_traj_segments, downsample_all), axis=0)
        self.size = len(trajs)

    def get_paired_batch(self, batch_size=128):
        traj_sample_index = np.random.choice(np.arange(self.size), size=batch_size, replace=False)

        traj_candidate = self.all_traj_segments[traj_sample_index]        # (batch, subsamples, steps, 18dims)
        L = traj_candidate.shape[-2]
        H_in = traj_candidate.shape[-1]

        # select transform
        view_1_index = np.random.randint(traj_candidate.shape[1], size=batch_size)
        # view_1_batch = (batch, steps, 18dims)
        view_1_batch = np.empty((0, L, H_in))
        for i in range(batch_size):
            view_1 = self.all_traj_segments[traj_sample_index[i], view_1_index[i], :, :]# (steps, 18dims)
            view_1_batch = np.concatenate((view_1_batch, np.expand_dims(view_1, axis=0)), axis=0)

        # select transform
        view_2_index = np.random.randint(traj_candidate.shape[1], size=batch_size)
        # view_1_batch = (batch, steps, 18dims)
        view_2_batch = np.empty((0, L, H_in))
        for i in range(batch_size):
            view_2 = self.all_traj_segments[traj_sample_index[i], view_2_index[i], :, :]# (1, 1, steps, 18dims)
            view_2_batch = np.concatenate((view_2_batch, np.expand_dims(view_2, axis=0)), axis=0)
        # permute
        view_1_batch = torch.tensor(view_1_batch).permute((1,0,2))
        view_2_batch = torch.tensor(view_2_batch).permute((1,0,2))
        return view_1_batch, view_2_batch # return (L, B, H_in)


class loss_simclr(torch.nn.Module):
    def __init__(self):
        super(loss_simclr, self).__init__()

    def forward(self, feat_1, feat_2, batch_size, temperature=0.05):
        # 分母 ：X * X.T，再去掉对角线值，(-1列)，可以看成它与除了这行外的其他行都进行了点积运算（包括feat_1和feat_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括feat_1和feat_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]
        out = torch.cat([feat_1, feat_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(feat_1 * feat_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
