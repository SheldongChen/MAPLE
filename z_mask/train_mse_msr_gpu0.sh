GPUs='0'
depth_decoder=8
mkdir output/msr_action_mae_10
CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_10' --data-meta './datasets/msr/msr_1.list'  --data-meta-unlabel  './datasets/msr/msr_1_unlabel.list' --loss-alpha 1.0 | tee output/msr_action_mae_10/ouput_msr_1_alpha10.log

CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_10' --data-meta './datasets/msr/msr_2.list'  --data-meta-unlabel  './datasets/msr/msr_2_unlabel.list' --loss-alpha 1.0 | tee output/msr_action_mae_10/ouput_msr_2_alpha10.log

CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_10' --data-meta './datasets/msr/msr_3.list'  --data-meta-unlabel  './datasets/msr/msr_3_unlabel.list' --loss-alpha 1.0 | tee output/msr_action_mae_10/ouput_msr_3_alpha10.log

CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_10' --data-meta './datasets/msr/msr_4.list'  --data-meta-unlabel  './datasets/msr/msr_4_unlabel.list' --loss-alpha 1.0 | tee output/msr_action_mae_10/ouput_msr_4_alpha10.log

CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_10' --data-meta './datasets/msr/msr_5.list'  --data-meta-unlabel  './datasets/msr/msr_5_unlabel.list' --loss-alpha 1.0 | tee output/msr_action_mae_10/ouput_msr_5_alpha10.log