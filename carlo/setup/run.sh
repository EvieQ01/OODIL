export CUDA_VISIBLE_DEVICES=0

# ################### Driving  #################################
################ Phase 1 ##########################
# Contrastive clustering.
mode=simclr_dcn
b=128
beta=0.01
clu=10
python main_contrastive1.py  \
        --demo_files  \
                '../demo/Maxspeed_1.0_[0.5, 0.25]_all_batch00.pkl'  \
                '../demo/Maxspeed_1.0_[0.1, 0.5]_all_batch00.pkl' \
                ../demo/Maxspeed_5.0_all_batch00.pkl \
        --env-name Continuous-v104025 --downsample_stride 15 --simclr_warmup 200  \
        --ratio .1 .5 .5 --mode ${mode}  --batch-size ${b} --temperature 1. --max_iteration 2000 --n-clusters $clu --beta $beta --seed 2 --dump


################ Phase 2 ##########################
# Construct transferability on each mode.
mode=simclr_dcn
for idx in {0..9}
do        
        python main_gail_transfer2.py \
        --env-name Continuous-v104025 \
        --demo_files '../demo/Continuous-v104025/re_split_simclr_dcn_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride15-ratio[0.1, 0.5, 0.5]-k10.pkl' \
        --ratio 1.  --batch-size 15000 \
        --init_range 0 60 \
        --begin-index ${idx} \
        --eval-interval 5 --num-epochs 2000  --mode ${mode} --eval_epochs 100
done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm
python main_gail_discri3.py \
        --env-name Continuous-v104025 \
        --demo_files '../demo/Continuous-v104025/re_split_simclr_dcn_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride15-ratio[0.1, 0.5, 0.5]-k10.pkl' \
        --cluster_list 0 1 2 3 4 5 6 7 8 9 \
        --ratio 1.  --batch-size 15000 \
                --init_range 0 60 \
        --eval-interval 5 --num-epochs 2000  --mode simclr_dcn --eval_epochs 100 \
        --feasibility_model \
                'log/Continuous-v104025/2GAIL_feas_simclr_dcn_ratio_[[1.0]]/checkpoints/seed_1111_gail_model_begin_index_0.pth' 
