#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
################ Phase 1 ##########################
# Contrastive clustering.
# resplit
# v13
    python main_contrastive1.py --env-name disabledpanda-v13 --demo_files \
                    'disabledpanda-v13/size100000_panda_init2.0_new.pth' \
                    'disabledpanda-v1/size100000_panda_init2.0_new.pth' \
                    'panda-v0/size100000_panda_init2.0_new.pth' \
                                                --ratio 1.0 1.0 1.0 1.0 --out_type cos --n-clusters 5 \
                            --temperature .3 --beta 0.01 --downsample_stride 15 --seed 0 --file_name from_13_1_0_to13_new_split0.pth --dump
# logs/disabledpanda-v13/resplit-dataset-DCN/simclr_5cluster/source-None-temperature0.3-beta-0.01-batch128-stride15-ratio[1.0, 1.0, 1.0, 1.0]-savefrom_13_1_0_to13_new_split0.pth/checkpoints/seed_0_rnn_cos_DCN_model.pth

################ Phase 2 ##########################
# Construct transferability on each mode.
for id in {0..4}
    do
    python  main_gail_transfer2.py --root_path 'disabledpanda-v13' --buffer 'from_13_1_0_to13_new_split0.pth' --env_id disabledpanda-v13 \
    --init_range 2.0 --begin_idx $id --mode resplit_simclr \
    --epoch_ppo 2 --epoch_disc 5 --use_minibatch --cuda
    done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
# change the --disc_model_path according to your own path
    python  main_gail_discri3.py --buffers \
                'disabledpanda-v13/from_13_1_0_to13_new_split0.pth' \
                --cluster_list 0 1 2 3 4 \
                 --env_id disabledpanda-v13 --init_range 2.0 --mode resplit_simclr \
                --source_str from_13_1_0  --seed 2 --cuda  --epoch_ppo 2 --epoch_disc 5 --use_minibatch --use_vectorenv \
                 --disc_model_paths \
                 logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221103-0858-cluster-4/model/step15000000/gail_disc.pth \
                logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221103-0859-cluster-3/model/step15000000/gail_disc.pth \
                logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221103-0900-cluster-2/model/step15000000/gail_disc.pth \
                logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221105-0205-cluster-1/model/step15000000/gail_disc.pth \
                logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221105-0205-cluster-0/model/step15000000/gail_disc.pth 
    
