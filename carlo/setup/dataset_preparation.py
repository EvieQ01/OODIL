import numpy as np
import pickle
import os

def load_demos(demo_files, ratios, num_inputs, args):
    r'''
    returns:
    state_files: (domain, traj_per_dom, step - 1, num_inputs)
    traj: (traj, step, num_inputs)
    '''
    state_files = []
    trajs = []
    traj_traj_id = []
    traj_id = 0
    pair_traj_id = []
    init_obs = []
    all_mapping = None
    count_of_traj = []
    for i in range(len(demo_files)):
        state_pairs = []
        demo_file = demo_files[i]
        if 'demo' not in demo_file:
            # add root path
            demo_file = os.path.join(args.dataset, demo_file)
        try:
            raw_demos = pickle.load(open(demo_file, 'rb'))
        except:
            print("empty cluster[loading fail]: ", demo_file)
            continue
        use_num = int(len(raw_demos['obs'])*ratios[i])
        current_state = raw_demos['obs'][0:use_num]

        print("=> use demo {}:".format(demo_files[i]), len(current_state))
        count_of_traj.append(len(current_state))
        trajs += [np.array(traj)[:, :num_inputs] for traj in current_state]# leave out the last obs dim
        # 0 0 0 ..., 1 1 1, ..., 2,2,2, ..., 200, 200, 200
        traj_traj_id += [i]*len(current_state)
        for j in range(len(current_state)):
            current_state[j] = np.array(current_state[j])[:, :num_inputs]# leave out the last obs dim
            state_pairs.append(np.concatenate([np.array(current_state[j][:-1, :]), np.array(current_state[j][1:, :])], axis=1))
            pair_traj_id.append(np.array([traj_id]*np.array(current_state[j]).shape[0]))
            traj_id += 1
        state_files.append(np.concatenate(state_pairs, axis=0))
    if len(state_files) == 0:
        return [[]]
    else:
        return state_files, trajs, np.concatenate(pair_traj_id, axis=0), np.array(traj_traj_id), init_obs, all_mapping, count_of_traj
