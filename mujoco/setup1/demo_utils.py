import pickle
import numpy as np

def load_demos(demo_files, ratios, begin_index=None, cluster_list=None):
    state_files = []
    trajs = []
    traj_traj_id = []
    traj_id = 0
    pair_traj_id = []
    count_of_traj = []
    demo_file_count = len(cluster_list) if cluster_list is not None else len(demo_files)
    clu_id = 0
    for i in range(demo_file_count):
        state_pairs = []
        # change according to begin_index
        if cluster_list is not None:
            demo_file = demo_files[0].replace('re_split_simclr_0', f're_split_simclr_{cluster_list[i]}')
            print('=>Warning: use demo_file(according to begin_index): ', demo_file)
        else:
            demo_file = demo_files[0].replace('re_split_simclr_0', f're_split_simclr_{begin_index}')
            print('=>Warning: use demo_file(according to begin_index): ', demo_file)
        try:
            raw_demos = pickle.load(open(demo_file, 'rb'))
        except:
            print("empty cluster[loading fail]: ", demo_file)
            continue
        use_num = int(len(raw_demos['obs'])*ratios[i])
        current_state = raw_demos['obs'][0:use_num]
        # s_t0, s_t1, s_t2, ...
        print("=> use demo {}:".format(demo_file), len(current_state))
        count_of_traj.append(len(current_state))
        trajs += [np.array(traj) for traj in current_state]
        # 0 0 0 ..., 1 1 1, ..., 2,2,2, ..., 200, 200, 200
        traj_traj_id += [clu_id]*len(current_state)
        clu_id += 1
        for j in range(len(current_state)):
            current_state[j] = np.array(current_state[j])
            state_pairs.append(np.concatenate([np.array(current_state[j][:-1, :]), np.array(current_state[j][1:, :])], axis=1))
            pair_traj_id.append(np.array([traj_id]*np.array(current_state[j]).shape[0]))
            traj_id += 1
        state_files.append(np.concatenate(state_pairs, axis=0))
    if len(state_files) == 0:
        return [[]]
    else:
        return state_files, trajs, np.concatenate(pair_traj_id, axis=0), np.array(traj_traj_id), count_of_traj

