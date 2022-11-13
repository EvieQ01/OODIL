export CUDA_VISIBLE_DEVICES=0 # we only use cuda for Phase 1 here

# ################### Hopper ###################################################
################ Phase 1 ##########################
# Contrastive clustering.
python main_contrastive1.py  \
	--demo_files ../demo/hopper_15.0/batch_00.pkl ../demo/hopper_9.8/batch_00.pkl  \
		../demo/hopper_2.0/batch_00.pkl ../demo/hopper_1.0/batch_00.pkl \
	--xml hopper_12.0.xml  \
	--env-name CustomHopper-v0 --batch-size 128 --temperature 1. --lamda 0.01 --dist_type cos \
	--ratio .02 .02 .5 .5   --max_iteration 2000 --simclr_warmup 200 --seed 0 \
	--n_clusters 5 --dump

################ Phase 2 ##########################
# Construct transferability on each mode.
mode=resplit-simclr
xml=12.0 
for idx in {0..4}
do
	python main_gail_transfer2.py  \
	--xml hopper_${xml}.xml  \
	--env-name CustomHopper-v0 --begin-index  ${idx} \
	--demo_files 'log/CustomHopper-v0/re_split_simclr_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride20-ratio[0.02, 0.02, 0.5, 0.5]-N5.pkl' \
	--ratios 1. --mode ${mode}  --eval-interval 5 --num-epochs 2000
done

################ Phase 3 ##########################
# Used the adversarial-based transferability to conduct the final imitation algorithm.
python main_gail_disc3.py \
	--demo_files 'log/CustomHopper-v0/re_split_simclr_0_DCN_batch_00_temperature1.0-beta-0.01-batch128-stride20-ratio[0.02, 0.02, 0.5, 0.5]-N5.pkl' \
	--xml hopper_${xml}.xml  \
	--env-name CustomHopper-v0  \
	--ratios 1. 1. 1. 1. 1. --mode ${mode}  --eval-interval 10 --num-epochs 2000 --seed 1111 \
	--cluster_list 0 1 2 3 4 \
	--feasibility_model \
		'log/CustomHopper-v0/resplit-simclr/target-hopper_12.0_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_0.pth' \
		'log/CustomHopper-v0/resplit-simclr/target-hopper_12.0_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_1.pth' \
		'log/CustomHopper-v0/resplit-simclr/target-hopper_12.0_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_2.pth' \
		'log/CustomHopper-v0/resplit-simclr/target-hopper_12.0_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_3.pth' \
		'log/CustomHopper-v0/resplit-simclr/target-hopper_12.0_ratio_[1.0]/checkpoints/seed_1111_gail_model_begin_index_4.pth' \

