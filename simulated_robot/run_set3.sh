export CUDA_VISIBLE_DEVICES=0
################ Phase 1 ##########################
# Contrastive clustering.
python main_contrastive1.py --env-name disabledpanda-v13 --demo_files 'buffers/disabledpanda-v4/size100000_panda.pth' \
                                            'buffers/disabledpanda-v6/size100000_panda.pth' \
                                            'buffers/disabledpanda-v1/size1000000_panda_init0.5.pth' \
                        --ratio 1.0 1.0 1.0 1.0 --dist_type cos --seed 1 \
                        --temperature 1. --beta 0.01 --downsample_stride 15  --file_name from_4_6_1_split0.pth --dump #--load_only 

################ Phase 2 ##########################
# Construct transferability on each mode.
for id in {0..9}
    do
        python  main_gail_transfer2.py --root_path 'disabledpanda-v13' --buffer 'from_4_6_1_split0.pth' --env_id disabledpanda-v13 --init_range 10.0 --begin_idx ${id} --mode resplit_simclr --light_obj
    done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
python  main_gail_discri3.py  --buffers 'disabledpanda-v13/from_4_6_1_split0.pth' --env_id disabledpanda-v13  --init_range 10.0 --mode resplit_simclr \
            --seed 0 --light_obj --hidden_units 100  \
            --cluster_list 0 1 2 3 4 5 6 7 \
            --disc_model_paths \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-0259-cluster-0/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-0259-cluster-1/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-0407-cluster-2/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-0407-cluster-3/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-1529-cluster-4/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-1529-cluster-5/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-1529-cluster-6/model/step30000000/gail_disc.pth' \
                'logs/disabledpanda-v13/gailfo/resplit_simclr-seed0-20221108-1529-cluster-7/model/step30000000/gail_disc.pth'  

python  main_gail_discri3.py  --buffers 'disabledpanda-v6/size100000_panda.pth' --env_id disabledpanda-v13  --init_range 2.0 --mode baseline \
            --seed 1 --light_obj --hidden_units 100  --render

                