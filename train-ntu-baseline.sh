mkdir ./best_model/ntu

python train-ntu-twotrans.py  --data-meta ./datasets/ntu/ntu60_half_cs_033.list | tee  output/ntu_twotrans/output_baseline_cs_033.log
cp output/ntu_twotrans/model_best.pth ./best_model/ntu/model_best_033.pth

python train-ntu-twotrans.py  --data-meta ./datasets/ntu/ntu60_half_cs_066.list | tee  output/ntu_twotrans/output_baseline_cs_066.log
cp output/ntu_twotrans/model_best.pth ./best_model/ntu/model_best_066.pth

python train-ntu-twotrans.py  --data-meta ./datasets/ntu/ntu60_half_cs_132.list | tee  output/ntu_twotrans/output_baseline_cs_132.log
cp output/ntu_twotrans/model_best.pth ./best_model/ntu/model_best_132.pth

python train-ntu-twotrans.py  --data-meta ./datasets/ntu/ntu60_half_cs_198.list | tee  output/ntu_twotrans/output_baseline_cs_198.log
cp output/ntu_twotrans/model_best.pth ./best_model/ntu/model_best_198.pth

python train-ntu-twotrans.py  --data-meta ./datasets/ntu/ntu60_half_cs_264.list | tee  output/ntu_twotrans/output_baseline_cs_264.log
cp output/ntu_twotrans/model_best.pth ./best_model/ntu/model_best_264.pth



