
mkdir output/ntu_twotrans_mae_alpha02_resume_decoder_entmin

python z_mask/train-ntu-twotrans.py --resume output_ntu/entmin/vat_eps01_alpha02_only_ul_False_resume_EntMin/vat_033/model_best.pth --mask-length 9 --decoder-depth-t 2 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder_entmin' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/ouput_ntu60_cs_033_alpha02_decoder2.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best_033_decoder2.pth

python z_mask/train-ntu-twotrans.py --resume output_ntu/entmin/vat_eps01_alpha02_only_ul_False_resume_EntMin/vat_033/model_best.pth --mask-length 9 --decoder-depth-t 4 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder_entmin' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/ouput_ntu60_cs_033_alpha02_decoder4.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best_033_decoder4.pth

python z_mask/train-ntu-twotrans.py --resume output_ntu/entmin/vat_eps01_alpha02_only_ul_False_resume_EntMin/vat_033/model_best.pth --mask-length 9 --decoder-depth-t 6 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder_entmin' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/ouput_ntu60_cs_033_alpha02_decoder6.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best_033_decoder6.pth



python z_mask/train-ntu-twotrans.py --resume output_ntu/entmin/vat_eps01_alpha02_only_ul_False_resume_EntMin/vat_033/model_best.pth --mask-length 9 --decoder-depth-t 10 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder_entmin' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/ouput_ntu60_cs_033_alpha02_decoder10.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best_033_decoder10.pth

python z_mask/train-ntu-twotrans.py --resume output_ntu/entmin/vat_eps01_alpha02_only_ul_False_resume_EntMin/vat_033/model_best.pth --mask-length 9 --decoder-depth-t 12 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder_entmin' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/ouput_ntu60_cs_033_alpha02_decoder12.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder_entmin/model_best_033_decoder12.pth