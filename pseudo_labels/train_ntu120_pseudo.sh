mkdir ./output/ntu120

echo 5
python pseudo_labels/train-ntu-120-twotrans.py  --loss-alpha  0.2  --lr  0.001  --data-meta './datasets/ntu/ntu120_half_cs_026.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_026_unlabel.list' --resume './best_model/ntu120/model_best_026.pth' --output-dir 'output/ntu120_pseudo_026' | tee ./output/ntu120/output_4GPU_xyz_4_t_3_ntu_026_pseudo.log

echo 10
python pseudo_labels/train-ntu-120-twotrans.py  --loss-alpha  0.2  --lr  0.001  --data-meta './datasets/ntu/ntu120_half_cs_053.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_053_unlabel.list' --resume './best_model/ntu120/model_best_053.pth' --output-dir 'output/ntu120_pseudo_053' | tee ./output/ntu120/output_4GPU_xyz_4_t_3_ntu_053_pseudo.log

echo 20
python pseudo_labels/train-ntu-120-twotrans.py  --loss-alpha  0.2  --lr  0.001  --data-meta './datasets/ntu/ntu120_half_cs_106.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_106_unlabel.list' --resume './best_model/ntu120/model_best_106.pth' --output-dir 'output/ntu120_pseudo_106' | tee ./output/ntu120/output_4GPU_xyz_4_t_3_ntu_106_pseudo.log

echo 30
python pseudo_labels/train-ntu-120-twotrans.py  --loss-alpha  0.2  --lr  0.001  --data-meta './datasets/ntu/ntu120_half_cs_159.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_159_unlabel.list' --resume './best_model/ntu120/model_best_159.pth'  --output-dir 'output/ntu120_pseudo_159' | tee ./output/ntu120/output_4GPU_xyz_4_t_3_ntu_159_pseudo.log

echo 40
python pseudo_labels/train-ntu-120-twotrans.py  --loss-alpha  0.2  --lr  0.001  --data-meta './datasets/ntu/ntu120_half_cs_212.list'  --data-meta-unlabel  './datasets/ntu/ntu120_half_cs_212_unlabel.list' --resume './best_model/ntu120/model_best_212.pth'  --output-dir 'output/ntu120_pseudo_212' | tee ./output/ntu120/output_4GPU_xyz_4_t_3_ntu_212_pseudo.log



