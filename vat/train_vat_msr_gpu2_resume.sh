#--vat-only-ul
mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin

GPUs='1'

mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_1

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --lr-warmup-epochs 5  --resume './best_model/msr_action_two/model_best_1.pth'  --data-meta './datasets/msr/msr_1.list'  --data-meta-unlabel  './datasets/msr/msr_1_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_1' --vat-eps 0.1 --vat-alpha 1.0 --vat-only-ul | tee ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_1/output_1GPU_xyz_4_t_3_msr_1_vat_resume.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_2

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --lr-warmup-epochs 5  --resume './best_model/msr_action_two/model_best_2.pth'  --data-meta './datasets/msr/msr_2.list'  --data-meta-unlabel  './datasets/msr/msr_2_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_2' --vat-eps 0.1 --vat-alpha 1.0 --vat-only-ul | tee ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_2/output_1GPU_xyz_4_t_3_msr_2_vat_resume.log


mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_3

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --lr-warmup-epochs 5  --resume './best_model/msr_action_two/model_best_3.pth'  --data-meta './datasets/msr/msr_3.list'  --data-meta-unlabel  './datasets/msr/msr_3_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_3' --vat-eps 0.1 --vat-alpha 1.0 --vat-only-ul | tee ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_3/output_1GPU_xyz_4_t_3_msr_3_vat_resume.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_4

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --lr-warmup-epochs 5  --resume './best_model/msr_action_two/model_best_4.pth'  --data-meta './datasets/msr/msr_4.list'  --data-meta-unlabel  './datasets/msr/msr_4_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_4' --vat-eps 0.1 --vat-alpha 1.0 --vat-only-ul | tee ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_4/output_1GPU_xyz_4_t_3_msr_4_vat_resume.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_5

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --lr-warmup-epochs 5  --resume './best_model/msr_action_two/model_best_5.pth'  --data-meta './datasets/msr/msr_5.list'  --data-meta-unlabel  './datasets/msr/msr_5_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_5' --vat-eps 0.1 --vat-alpha 1.0 --vat-only-ul | tee ./best_model/msr_action_two/vat_eps2_alpha1_only_ul_True_EntMin/vat_5/output_1GPU_xyz_4_t_3_msr_5_vat_resume.log



