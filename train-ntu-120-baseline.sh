mkdir ./best_model/ntu120

python train-ntu-120-twotrans.py  --data-meta ./datasets/ntu/ntu120_half_cs_212.list | tee  output/ntu120_twotrans/output_depth5_32bs_relu_r01_data_transfrom_cs_212.log

cp ./output/ntu120_twotrans/model_best.pth ./best_model/ntu120/model_best_212.pth

python train-ntu-120-twotrans.py  --data-meta ./datasets/ntu/ntu120_half_cs_159.list | tee  output/ntu120_twotrans/output_depth5_32bs_relu_r01_data_transfrom_cs_159.log

cp ./output/ntu120_twotrans/model_best.pth ./best_model/ntu120/model_best_159.pth

python train-ntu-120-twotrans.py  --data-meta ./datasets/ntu/ntu120_half_cs_106.list | tee  output/ntu120_twotrans/output_depth5_32bs_relu_r01_data_transfrom_cs_106.log

cp ./output/ntu120_twotrans/model_best.pth ./best_model/ntu120/model_best_106.pth

python train-ntu-120-twotrans.py  --data-meta ./datasets/ntu/ntu120_half_cs_053.list | tee  output/ntu120_twotrans/output_depth5_32bs_relu_r01_data_transfrom_cs_053.log

cp ./output/ntu120_twotrans/model_best.pth ./best_model/ntu120/model_best_053.pth

python train-ntu-120-twotrans.py  --data-meta ./datasets/ntu/ntu120_half_cs_026.list | tee  output/ntu120_twotrans/output_depth5_32bs_relu_r01_data_transfrom_cs_026.log

cp ./output/ntu120_twotrans/model_best.pth ./best_model/ntu120/model_best_026.pth









