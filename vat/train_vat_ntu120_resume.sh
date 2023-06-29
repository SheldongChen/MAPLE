#--vat-only-ul
mkdir ./best_model/ntu120
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume


#026
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_026

python vat/train-ntu-120-twotrans-entmin.py    --lr 0.01  --vat-eps 0.2  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './output_ntu120/best_model/two_trans/model_best_026.pth'   --data-meta './datasets/ntu/ntu120_half_cs_026.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_026_unlabel.list'  --output-dir './best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_026'  --vat-alpha 0.6  | tee ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_026/output_4GPU_xyz_4_t_3_ntu_026_vat_resume.log

#053
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_053

python vat/train-ntu-120-twotrans-entmin.py    --lr 0.01  --vat-eps 0.2  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './output_ntu120/best_model/two_trans/model_best_053.pth'   --data-meta './datasets/ntu/ntu120_half_cs_053.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_053_unlabel.list'  --output-dir './best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_053'  --vat-alpha 0.6  | tee ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_053/output_4GPU_xyz_4_t_3_ntu_053_vat_resume.log

#106
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_106

python vat/train-ntu-120-twotrans-entmin.py    --lr 0.01  --vat-eps 0.2  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './output_ntu120/best_model/two_trans/model_best_106.pth'   --data-meta './datasets/ntu/ntu120_half_cs_106.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_106_unlabel.list'  --output-dir './best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_106'  --vat-alpha 0.6  | tee ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_106/output_4GPU_xyz_4_t_3_ntu_106_vat_resume.log

#159
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_159

python vat/train-ntu-120-twotrans-entmin.py    --lr 0.01  --vat-eps 0.2  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './output_ntu120/best_model/two_trans/model_best_159.pth'   --data-meta './datasets/ntu/ntu120_half_cs_159.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_159_unlabel.list'  --output-dir './best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_159'  --vat-alpha 0.6  | tee ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_159/output_4GPU_xyz_4_t_3_ntu_159_vat_resume.log

#212
mkdir ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_212

python vat/train-ntu-120-twotrans-entmin.py    --lr 0.01  --vat-eps 0.2  --lr-warmup-epochs 1  --lr-milestones 10 15  --resume './output_ntu120/best_model/two_trans/model_best_212.pth'   --data-meta './datasets/ntu/ntu120_half_cs_212.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_212_unlabel.list'  --output-dir './best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_212'  --vat-alpha 0.6  | tee ./best_model/ntu120/vat_eps01_alpha04_only_ul_False_resume/vat_212/output_4GPU_xyz_4_t_3_ntu_212_vat_resume.log