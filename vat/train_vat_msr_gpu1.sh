#--vat-only-ul
mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin

GPUs='1'

mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_1

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --data-meta './datasets/msr/msr_1.list'  --data-meta-unlabel  './datasets/msr/msr_1_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_1' --vat-eps 0.1 --vat-alpha 0.5 | tee ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_1/output_1GPU_xyz_4_t_3_msr_1_vat.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_2

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --data-meta './datasets/msr/msr_2.list'  --data-meta-unlabel  './datasets/msr/msr_2_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_2' --vat-eps 0.1 --vat-alpha 0.5 | tee ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_2/output_1GPU_xyz_4_t_3_msr_2_vat.log


mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_3

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --data-meta './datasets/msr/msr_3.list'  --data-meta-unlabel  './datasets/msr/msr_3_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_3' --vat-eps 0.1 --vat-alpha 0.5 | tee ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_3/output_1GPU_xyz_4_t_3_msr_3_vat.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_4

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --data-meta './datasets/msr/msr_4.list'  --data-meta-unlabel  './datasets/msr/msr_4_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_4' --vat-eps 0.1 --vat-alpha 0.5 | tee ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_4/output_1GPU_xyz_4_t_3_msr_4_vat.log

mkdir ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_5

CUDA_VISIBLE_DEVICES=$GPUs python vat/train-msr-twotrans-entmin.py  --vat-EntMin  --data-meta './datasets/msr/msr_5.list'  --data-meta-unlabel  './datasets/msr/msr_5_unlabel.list'  --output-dir './best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_5' --vat-eps 0.1 --vat-alpha 0.5 | tee ./best_model/msr_action_two/vat_eps2_alpha05_only_ul_False_EntMin/vat_5/output_1GPU_xyz_4_t_3_msr_5_vat.log



