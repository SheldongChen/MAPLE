
mkdir output/ntu_twotrans_mae_alpha02_resume_decoder

python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 2 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder2.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder2.pth

python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 4 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder4.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder4.pth

python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 6 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder6.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder6.pth



python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 10 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder10.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder10.pth

python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 12 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder12.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder12.pth

python z_mask/train-ntu-twotrans.py --resume best_model/ntu/model_best_033.pth --decoder-depth-t 14 --output-dir 'output/ntu_twotrans_mae_alpha02_resume_decoder' --data-meta 'datasets/ntu/ntu60_half_cs_033.list'  --data-meta-unlabel  'datasets/ntu/ntu60_half_cs_033_unlabel.list' --loss-alpha 0.2 | tee output/ntu_twotrans_mae_alpha02_resume_decoder/ouput_ntu60_cs_033_alpha02_decoder14.log

cp output/ntu_twotrans_mae_alpha02_resume_decoder/model_best.pth  output/ntu_twotrans_mae_alpha02_resume_decoder/model_best_033_decoder14.pth