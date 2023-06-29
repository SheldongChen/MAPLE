mkdir ./best_model/msr_action_two

echo 5
CUDA_VISIBLE_DEVICES='0' python pseudo_labels/train-msr-twotrans.py  --data-meta './datasets/msr/msr_1.list'  --data-meta-unlabel  './datasets/msr/msr_1_unlabel.list' --resume './best_model/msr_action_two/model_best_1.pth' --output-dir 'output/msr_action_pseudo_1' | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_1_pseudo.log

echo 10
CUDA_VISIBLE_DEVICES='1' python pseudo_labels/train-msr-twotrans.py  --data-meta './datasets/msr/msr_2.list'  --data-meta-unlabel  './datasets/msr/msr_2_unlabel.list' --resume './best_model/msr_action_two/model_best_2.pth' --output-dir 'output/msr_action_pseudo_2' | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_2_pseudo.log

echo 20
CUDA_VISIBLE_DEVICES='2' python pseudo_labels/train-msr-twotrans.py  --data-meta './datasets/msr/msr_3.list'  --data-meta-unlabel  './datasets/msr/msr_3_unlabel.list' --resume './best_model/msr_action_two/model_best_3.pth' --output-dir 'output/msr_action_pseudo_3' | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_3_pseudo.log

echo 30
CUDA_VISIBLE_DEVICES='3' python pseudo_labels/train-msr-twotrans.py  --data-meta './datasets/msr/msr_4.list'  --data-meta-unlabel  './datasets/msr/msr_4_unlabel.list' --resume './best_model/msr_action_two/model_best_4.pth'  --output-dir 'output/msr_action_pseudo_4' | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_4_pseudo.log

echo 40
CUDA_VISIBLE_DEVICES='3' python pseudo_labels/train-msr-twotrans.py  --data-meta './datasets/msr/msr_5.list'  --data-meta-unlabel  './datasets/msr/msr_5_unlabel.list' --resume './best_model/msr_action_two/model_best_5.pth'  --output-dir 'output/msr_action_pseudo_5' | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_5_pseudo.log



