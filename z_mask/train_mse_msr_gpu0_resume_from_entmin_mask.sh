GPUs='0'
depth_decoder=8
mkdir output/msr_action_mae_05_resumefromentmin_mask


CUDA_VISIBLE_DEVICES=$GPUs python z_mask/train-msr-twotrans.py --mask-length 9 --decoder-depth-t $depth_decoder --output-dir 'output/msr_action_mae_05_resumefromentmin_mask' --data-meta './datasets/msr/msr_1.list'  --data-meta-unlabel  './datasets/msr/msr_1_unlabel.list' --loss-alpha 0.5 --resume 'output_msr/entmin/model_best_1.pth'  --lr-warmup-epochs 1 | tee output/msr_action_mae_05_resumefromentmin_mask/ouput_msr_1_alpha05_resume_mask9.log
