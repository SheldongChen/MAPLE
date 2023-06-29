#--vat-only-ul
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume


#033
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_033

python vat/train-ntu-twotrans-entmin.py   --lr 0.01  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './best_model/ntu/model_best_033.pth'   --data-meta './datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_033_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_033'  --vat-alpha 0.4  | tee ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_033/output_4GPU_xyz_4_t_3_ntu_033_vat_resume.log

#066
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_066

python vat/train-ntu-twotrans-entmin.py   --lr 0.01  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './best_model/ntu/model_best_066.pth'   --data-meta './datasets/ntu/ntu60_half_cs_066.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_066_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_066'  --vat-alpha 0.4  | tee ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_066/output_4GPU_xyz_4_t_3_ntu_066_vat_resume.log

#132
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_132

python vat/train-ntu-twotrans-entmin.py   --lr 0.01  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './best_model/ntu/model_best_132.pth'   --data-meta './datasets/ntu/ntu60_half_cs_132.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_132_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_132'  --vat-alpha 0.4  | tee ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_132/output_4GPU_xyz_4_t_3_ntu_132_vat_resume.log

#198
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_198

python vat/train-ntu-twotrans-entmin.py   --lr 0.01  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './best_model/ntu/model_best_198.pth'   --data-meta './datasets/ntu/ntu60_half_cs_198.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_198_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_198'  --vat-alpha 0.4  | tee ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_198/output_4GPU_xyz_4_t_3_ntu_198_vat_resume.log

#264
mkdir ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_264

python vat/train-ntu-twotrans-entmin.py   --lr 0.01  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './best_model/ntu/model_best_264.pth'   --data-meta './datasets/ntu/ntu60_half_cs_264.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_264_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_264'  --vat-alpha 0.4  | tee ./best_model/ntu/vat_eps01_alpha04_only_ul_False_resume/vat_264/output_4GPU_xyz_4_t_3_ntu_264_vat_resume.log