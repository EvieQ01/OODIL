export CUDA_VISIBLE_DEVICES=1 # we only use cuda for Phase 1 here

# ################### Walker  #################################
################ Phase 1 ##########################
# Contrastive clustering.
python main_contrastive1.py  \
    --demo_files ../demo/walker2d_24.8/batch_00.pkl \
            ../demo/walker2d_9.9/batch_00.pkl \
            ../demo/walker2d_3.9/batch_00.pkl \
            ../demo/walker2d_1.1/batch_00.pkl \
    --xml walker2d_19.9.xml  \
    --env-name CustomWalker2d-v0 --batch-size 128 --temperature 1. --lamda 0.01 --dist_type cos \
    --ratio .05 .05 .1 .1   --max_iteration 2000 --simclr_warmup 200 --seed 0 \
    --n_clusters 5 --dump


################ Phase 2 ##########################
# Construct transferability on each mode.
mode=resplit-simclr
xml=19.9
for idx in {0..4} 
    do
        python main_gail_transfer2.py  \
        --xml walker2d_${xml}.xml  \
        --env-name CustomWalker2d-v0 --begin-index  ${idx} \
        --demo_files 'log/CustomWalker2d-v0/re_split_simclr_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride20-ratio[0.05, 0.05, 0.1, 0.1]-N5.pkl' \
        --ratios 1. --mode ${mode}  --eval-interval 5 --num-epochs 2
    done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
python main_gail_disc3.py \
    --xml walker2d_${xml}.xml  \
    --env-name CustomWalker2d-v0 \
    --ratio 1. 1. 1. 1. 1. \
    --demo_files \
            'log/CustomWalker2d-v0/re_split_simclr_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride20-ratio[0.05, 0.05, 0.1, 0.1]-N5.pkl' \
            --cluster_list 0 1 2 3 4 --seed 0 \
            --eval-interval 5 --num-epochs 4000  --mode ${mode} \
    --feasibility_model \
        'log/CustomWalker2d-v0/resplit-simclr/target-walker2d_19.9_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_0.pth' \
        'log/CustomWalker2d-v0/resplit-simclr/target-walker2d_19.9_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_1.pth' \
        'log/CustomWalker2d-v0/resplit-simclr/target-walker2d_19.9_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_2.pth' \
        'log/CustomWalker2d-v0/resplit-simclr/target-walker2d_19.9_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_3.pth' \
        'log/CustomWalker2d-v0/resplit-simclr/target-walker2d_19.9_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_4.pth' 
