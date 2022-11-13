export CUDA_VISIBLE_DEVICES=0 # we only use cuda for Phase 1 here

# ################### HalfCheetah  #################################
################ Phase 1 ##########################
# Contrastive clustering.
python main_contrastive1.py  \
    --demo_files ../demo/half_cheetah_back_0.9/batch_00.pkl ../demo/half_cheetah_front_0.9/batch_00.pkl \
                    ../demo/half_cheetah_back_0.05/batch_00.pkl ../demo/half_cheetah_back_0.5/batch_00.pkl \
    --xml half_cheetah.xml --dist_type cos \
    --env-name CustomHalfCheetah-v0 --batch-size 128 --temperature .1 --lamda 0.01 \
    --ratio .2 .2 .5 .5   --max_iteration 2000 --simclr_warmup 200 --seed 0 \
    --n_clusters 5 --dump



################ Phase 2 ##########################
# Construct transferability on each mode.
mode=resplit-simclr
xml=half_cheetah
for idx in {0..4}
    do
        python main_gail_transfer2.py  \
            --demo_files 'log/CustomHalfCheetah-v0/re_split_simclr_0_DCN_batch_00_temperature0.1-beta-0.01-batch128-stride20-ratio[0.2, 0.2, 0.5, 0.5]-N5.pkl' \
            --xml ${xml}.xml  \
            --env-name CustomHalfCheetah-v0 --begin-index $idx  --batch-size 15000 \
            --ratios 1. --mode ${mode} --eval-interval 5 --num-epochs 2000
    done


################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
python main_gail_disc3.py  \
    --xml ${xml}.xml  \
    --env-name CustomHalfCheetah-v0 \
    --demo_files 'log/CustomHalfCheetah-v0/re_split_simclr_0_DCN_batch_00_temperature0.1-beta-0.01-batch128-stride20-ratio[0.2, 0.2, 0.5, 0.5]-N5.pkl' \
    --ratio 1. 1. 1. 1. 1.  \
    --cluster_list 0 1 2 3 4 \
    --eval-interval 5 --num-epochs 4000 --mode ${mode} --batch-size 15000 --seed 1111 \
    --feasibility_model \
        'log/CustomHalfCheetah-v0/resplit-simclr/target-half_cheetah_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_0.pth'  \
        'log/CustomHalfCheetah-v0/resplit-simclr/target-half_cheetah_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_1.pth'  \
        'log/CustomHalfCheetah-v0/resplit-simclr/target-half_cheetah_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_2.pth' \
        'log/CustomHalfCheetah-v0/resplit-simclr/target-half_cheetah_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_3.pth' \
        'log/CustomHalfCheetah-v0/resplit-simclr/target-half_cheetah_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_4.pth' 
