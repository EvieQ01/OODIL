#!/bin/bash
# v134
export CUDA_VISIBLE_DEVICES=2
################ Phase 1 ##########################
# Contrastive clustering.
# resplit
    python main_contrastive1.py --env-name disabledpanda-v134 --demo_files \
                    'disabledpanda-v1346/size100000_panda_init2.0_new.pth' \
                    'disabledpanda-v13/size100000_panda_init2.0_goal0.81_new.pth' \
                    'disabledpanda-v4/size100000_panda_init2.0_new.pth' \
                                                --ratio 1.0 1.0 1.0 1.0 --out_type cos --n-clusters 5 \
                            --temperature .3 --beta 0.01 --downsample_stride 15 --seed 0 --file_name from_1346_13_4_to134_new_split0.pth --dump
################ Phase 2 ##########################
# Construct transferability on each mode.
for id in {0..4}
do
    python  main_gail_transfer2.py --root_path 'disabledpanda-v134' --buffer 'from_1346_13_4_to134_new_split0.pth' --env_id disabledpanda-v134 \
    --init_range 2.0 --begin_idx $id --mode resplit_simclr \
    --epoch_ppo 10 --epoch_disc 20 --use_minibatch --cuda
done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
# change the --disc_model_path according to your own path
    python  main_gail_discri3.py --buffers \
                'disabledpanda-v134/from_1346_13_4_to134_new_split0.pth' \
                --cluster_list 0 1 2 3 4 \
                --env_id disabledpanda-v134 --init_range 2.0 --mode resplit_simclr \
                --source_str from_1346_13_4  --seed 0 --cuda  --epoch_ppo 10 --epoch_disc 20 --use_minibatch --light_obj \
                --disc_model_paths \
                    logs/disabledpanda-v134/gailfo/resplit_simclr-seed0-20221029-0412-cluster-0/model/step14500000/gail_disc.pth \
                    logs/disabledpanda-v134/gailfo/resplit_simclr-seed0-20221029-0413-cluster-1/model/step14500000/gail_disc.pth \
                    logs/disabledpanda-v134/gailfo/resplit_simclr-seed0-20221030-0955-cluster-2/model/step14500000/gail_disc.pth \
                    logs/disabledpanda-v134/gailfo/resplit_simclr-seed0-20221030-0957-cluster-3/model/step14500000/gail_disc.pth \
                    logs/disabledpanda-v134/gailfo/resplit_simclr-seed0-20221030-0959-cluster-4/model/step14500000/gail_disc.pth \
                --use_vectorenv
