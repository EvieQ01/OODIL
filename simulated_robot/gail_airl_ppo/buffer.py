from copy import deepcopy
import os
import pdb
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        self.traj_len = tmp['traj_len']#.clone().to(self.device)

        self.weight = None #torch or numpy?TODO
        # self.traj_len = []
    def sample(self, batch_size):
        # idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        # all shape as self.buffer_size

        all_idxes = np.arange(self.buffer_size)
        # if self.weight is not None:
        idxes = np.random.choice(all_idxes, size=batch_size, p=self.weight)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )
    def get_all_pairs_for_traj(self, traj_idx):
        '''
        get all pairs for single traj.
        '''
        begin_state_idx = 0
        for i in range(traj_idx):
            begin_state_idx += self.traj_len[i]
        return self.states[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
    def get_all_actions_for_traj(self, traj_idx):
        '''
        get all pairs for single traj.
        '''
        begin_state_idx = 0
        for i in range(traj_idx):
            begin_state_idx += self.traj_len[i]
        return self.actions[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
    def get_all_next_pairs_for_traj(self, traj_idx):
        '''
        get all pairs for single traj.
        '''
        begin_state_idx = 0
        for i in range(traj_idx):
            begin_state_idx += self.traj_len[i]
        return self.next_states[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
    
    def get_episode_states_all(self):
        data = []
        for i in range(len(self.traj_len)):
            data.append(self.get_all_pairs_for_traj(i))
        return data
    def get_episode_actions_all(self):
        data = []
        for i in range(len(self.traj_len)):
            data.append(self.get_all_actions_for_traj(i))
        return data


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.traj_len = []
        self.weight = torch.ones(self.buffer_size) / self.buffer_size
    
    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)
        # if done == 1.:
        #     self.traj_len.append(self._p)
    def append_multi(self, state, action, reward, done, next_state):
        """
        append multiple (s, a, r, done, s')
        
        """
        assert len(state) == len(action)
        assert len(action) == len(reward)
        assert len(reward) == len(done)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        done = np.array(done)
        next_state = np.array(next_state)
        for i in range(len(state)):
            self.append(state[i], action[i], reward[i], done[i], next_state[i])    
    
    def add_traj_length(self, length):
        self.traj_len.append(length)
        

    def remove(self, step_count):

        
        # self.states = self.states[:-step_count]
        # self.actions = self.actions[:-step_count]
        # self.rewards = self.rewards[:-step_count]
        # self.dones = self.dones[:-step_count]
        # self.next_states = self.next_states[:-step_count]

        self._p = (self._p - step_count) #% self.buffer_size
        self._n = min(self._n - step_count, self.buffer_size)

    def save(self, path, true_length=None):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if true_length is None:
            true_length = self.buffer_size
        torch.save({
            'state': self.states.clone().cpu()[:true_length],
            'action': self.actions.clone().cpu()[:true_length],
            'reward': self.rewards.clone().cpu()[:true_length],
            'done': self.dones.clone().cpu()[:true_length],
            'next_state': self.next_states.clone().cpu()[:true_length],
            'traj_len':self.traj_len
        }, path)
    def reweight_for_each_traj(self, weight):
        '''
        weight is shape as len(traj_len). a weight for each trajectory
        turn into self.weight. a weight for each pair.
        '''
        traj_ids = np.ones(self.buffer_size,dtype=int)
        begin = 0
        for traj_id in range(len(self.traj_len)):
            traj_ids[begin:begin + self.traj_len[traj_id]] = traj_id
            begin += self.traj_len[traj_id]
        
        self.weight = weight[traj_ids]
        self.weight = self.weight / (np.sum(self.weight + 0.0000001) )
        self.weight[-1] = 1 - np.sum(self.weight[:-1])


    def get_sub_buffer(self, traj_idx_list):

        traj_len = []
        for traj_idx in traj_idx_list:
            traj_len.append(self.traj_len[traj_idx])
        sub_buffer_size = sum(traj_len)
        
        sub_buffer = Buffer(buffer_size=sub_buffer_size, state_shape=self.states.shape[1:], action_shape=self.actions.shape[1:], device=self.device)
        for traj_idx in traj_idx_list:
            begin_state_idx = 0
            for i in range(traj_idx):
                begin_state_idx += self.traj_len[i] # here we get the begin_state_idx
            state = self.states[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
            action = self.actions[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
            reward = self.rewards[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
            next_state = self.next_states[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
            mask = self.dones[begin_state_idx:begin_state_idx + self.traj_len[traj_idx], :]
            sub_buffer.append_multi(state, action, reward, mask, next_state)
            sub_buffer.add_traj_length(self.traj_len[traj_idx])
        return sub_buffer
class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def append_vector(self, state, action, reward, done, log_pi, next_state):
        bs = state.shape[0]
        assert(self._p+bs <= self.total_size)
        
        self.states[self._p: self._p+bs].copy_(torch.from_numpy(state))
        self.actions[self._p: self._p+bs].copy_(torch.from_numpy(action))
        self.rewards[self._p: self._p+bs].copy_(torch.from_numpy(reward[:, None]))
        self.dones[self._p: self._p+bs].copy_(torch.from_numpy(done[:, None]))
        self.log_pis[self._p: self._p+bs].copy_(torch.from_numpy(log_pi))
        self.next_states[self._p: self._p+bs].copy_(torch.from_numpy(next_state))

        self._p = (self._p + state.shape[0]) % self.total_size
        self._n = min(self._n + state.shape[0], self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )


class ConcatStateBuffers(Buffer):
    def __init__(self, buffer_list):
        self._n = 0
        self._p = 0
        buffer_size_list = [bu.buffer_size for bu in buffer_list]
        self.buffer_size = sum(buffer_size_list)
        device = buffer_list[0].device
        self.device = device
        
        self.states = torch.empty(
            (0, *(buffer_list[0].states.shape[1:])), dtype=torch.float, device=device)
        # self.actions = torch.empty(
        #     (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (0, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (0, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (0 *(buffer_list[0].states.shape[1:])), dtype=torch.float, device=device)
        self.traj_len = []
        self.domain_traj_count = []
        for bu in buffer_list:
            self.traj_len += bu.traj_len
            self.domain_traj_count.append(len(bu.traj_len)) # count of traj per Buffer
            if self.states.shape[0] == 0:
            # add s,a,r,s'
                self.states = bu.states
                # self.actions = torch.empty(
                #     (buffer_size, *action_shape), dtype=torch.float, device=device)
                self.actions = bu.rewards
                self.rewards = bu.rewards
                self.dones = bu.dones
                self.next_states = bu.next_states
                continue

            # add s,a,r,s'
            self.states = deepcopy(torch.cat((self.states, bu.states), dim=0))
            # self.actions = torch.empty(
            #     (buffer_size, *action_shape), dtype=torch.float, device=device)
            self.actions = deepcopy(torch.cat((self.actions, bu.rewards), dim=0))
            self.rewards = deepcopy(torch.cat((self.rewards, bu.rewards), dim=0))
            self.dones = deepcopy(torch.cat((self.dones, bu.dones), dim=0))
            self.next_states = deepcopy(torch.cat((self.next_states, bu.next_states), dim=0))
        try:
            assert self.states.shape[0] == self.buffer_size
        except:
            raise ValueError('Concat buffersize mismatch!')
        self.weight = None