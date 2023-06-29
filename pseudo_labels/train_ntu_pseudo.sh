#mkdir ./best_model/ntu

echo 5
python pseudo_labels/train-ntu-twotrans.py  --data-meta './datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_033_unlabel.list' --resume './best_model/ntu/model_best_033.pth' --output-dir 'output/ntu_pseudo_033' | tee output/ntu/output_4GPU_xyz_4_t_3_ntu_033_pseudo.log

echo 10
python pseudo_labels/train-ntu-twotrans.py  --data-meta './datasets/ntu/ntu60_half_cs_066.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_066_unlabel.list' --resume './best_model/ntu/model_best_066.pth' --output-dir 'output/ntu_pseudo_066' | tee output/ntu/output_4GPU_xyz_4_t_3_ntu_066_pseudo.log

echo 20
python pseudo_labels/train-ntu-twotrans.py  --data-meta './datasets/ntu/ntu60_half_cs_132.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_132_unlabel.list' --resume './best_model/ntu/model_best_132.pth' --output-dir 'output/ntu_pseudo_132' | tee output/ntu/output_4GPU_xyz_4_t_3_ntu_132_pseudo.log

echo 30
python pseudo_labels/train-ntu-twotrans.py  --data-meta './datasets/ntu/ntu60_half_cs_198.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_198_unlabel.list' --resume './best_model/ntu/model_best_198.pth'  --output-dir 'output/ntu_pseudo_198' | tee output/ntu/output_4GPU_xyz_4_t_3_ntu_198_pseudo.log

echo 40
python pseudo_labels/train-ntu-twotrans.py  --data-meta './datasets/ntu/ntu60_half_cs_264.list'  --data-meta-unlabel  './datasets/ntu/ntu60_half_cs_264_unlabel.list' --resume './best_model/ntu/model_best_264.pth'  --output-dir 'output/ntu_pseudo_264' | tee output/ntu/output_4GPU_xyz_4_t_3_ntu_264_pseudo.log



