
depth_decoder=8
mkdir output/ntu120_twotrans_mae_alpha01_resume

python z_mask/train-ntu-120-twotrans.py --resume output_ntu120/best_model/two_trans/model_best_026.pth --decoder-depth-t $depth_decoder --output-dir 'output/ntu120_twotrans_mae_alpha01_resume' --data-meta 'datasets/ntu/ntu120_half_cs_026.list'  --data-meta-unlabel  'datasets/ntu/ntu120_half_cs_026_unlabel.list' --loss-alpha 0.1 | tee output/ntu120_twotrans_mae_alpha01_resume/ouput_ntu120_cs_026_alpha01.log

cp output/ntu120_twotrans_mae_alpha01_resume/model_best.pth  output/ntu120_twotrans_mae_alpha01_resume/model_best_026.pth

python z_mask/train-ntu-120-twotrans.py --resume output_ntu120/best_model/two_trans/model_best_053.pth --decoder-depth-t $depth_decoder --output-dir 'output/ntu120_twotrans_mae_alpha01_resume' --data-meta 'datasets/ntu/ntu120_half_cs_053.list'  --data-meta-unlabel  'datasets/ntu/ntu120_half_cs_053_unlabel.list' --loss-alpha 0.1 | tee output/ntu120_twotrans_mae_alpha01_resume/ouput_ntu120_cs_053_alpha01.log

cp output/ntu120_twotrans_mae_alpha01_resume/model_best.pth  output/ntu120_twotrans_mae_alpha01_resume/model_best_053.pth

python z_mask/train-ntu-120-twotrans.py --resume output_ntu120/best_model/two_trans/model_best_106.pth --decoder-depth-t $depth_decoder --output-dir 'output/ntu120_twotrans_mae_alpha01_resume' --data-meta 'datasets/ntu/ntu120_half_cs_106.list'  --data-meta-unlabel  'datasets/ntu/ntu120_half_cs_106_unlabel.list' --loss-alpha 0.1 | tee output/ntu120_twotrans_mae_alpha01_resume/ouput_ntu120_cs_106_alpha01.log

cp output/ntu120_twotrans_mae_alpha01_resume/model_best.pth  output/ntu120_twotrans_mae_alpha01_resume/model_best_106.pth

python z_mask/train-ntu-120-twotrans.py --resume output_ntu120/best_model/two_trans/model_best_159.pth --decoder-depth-t $depth_decoder --output-dir 'output/ntu120_twotrans_mae_alpha01_resume' --data-meta 'datasets/ntu/ntu120_half_cs_159.list'  --data-meta-unlabel  'datasets/ntu/ntu120_half_cs_159_unlabel.list' --loss-alpha 0.1 | tee output/ntu120_twotrans_mae_alpha01_resume/ouput_ntu120_cs_159_alpha01.log

cp output/ntu120_twotrans_mae_alpha01_resume/model_best.pth  output/ntu120_twotrans_mae_alpha01_resume/model_best_159.pth

python z_mask/train-ntu-120-twotrans.py --resume output_ntu120/best_model/two_trans/model_best_212.pth --decoder-depth-t $depth_decoder --output-dir 'output/ntu120_twotrans_mae_alpha01_resume' --data-meta 'datasets/ntu/ntu120_half_cs_212.list'  --data-meta-unlabel  'datasets/ntu/ntu120_half_cs_212_unlabel.list' --loss-alpha 0.1 | tee output/ntu120_twotrans_mae_alpha01_resume/ouput_ntu120_cs_212_alpha01.log

cp output/ntu120_twotrans_mae_alpha01_resume/model_best.pth  output/ntu120_twotrans_mae_alpha01_resume/model_best_212.pth