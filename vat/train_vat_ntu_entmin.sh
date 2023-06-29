#--vat-only-ul
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin

echo 15
#033
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_033

python vat/train-ntu-twotrans-entmin.py  --vat-EntMin   --data-meta './datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_033_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_033'  --vat-alpha 1.0  | tee ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_033/output_4GPU_xyz_4_t_3_ntu_033_vat.log

#066
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_066

python vat/train-ntu-twotrans-entmin.py  --vat-EntMin   --data-meta './datasets/ntu/ntu60_half_cs_066.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_066_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_066'  --vat-alpha 1.0  | tee ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_066/output_4GPU_xyz_4_t_3_ntu_066_vat.log

#132
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_132

python vat/train-ntu-twotrans-entmin.py  --vat-EntMin   --data-meta './datasets/ntu/ntu60_half_cs_132.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_132_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_132'  --vat-alpha 1.0  | tee ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_132/output_4GPU_xyz_4_t_3_ntu_132_vat.log

#198
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_198

python vat/train-ntu-twotrans-entmin.py  --vat-EntMin   --data-meta './datasets/ntu/ntu60_half_cs_198.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_198_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_198'  --vat-alpha 1.0  | tee ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_198/output_4GPU_xyz_4_t_3_ntu_198_vat.log

#264
mkdir ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_264

python vat/train-ntu-twotrans-entmin.py  --vat-EntMin   --data-meta './datasets/ntu/ntu60_half_cs_264.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_264_unlabel.list'  --output-dir './best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_264'  --vat-alpha 1.0  | tee ./best_model/ntu/vat_eps01_alpha1_only_ul_False_Entmin/vat_264/output_4GPU_xyz_4_t_3_ntu_264_vat.log