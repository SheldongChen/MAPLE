GPUs='0'

mkdir output/msr_action_two
mkdir ./best_model/msr_action_two

CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_1.list | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_1.log
cp output/msr_action_two/model_best.pth  ./best_model/msr_action_two/model_best_1.pth


CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_2.list | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_2.log
cp output/msr_action_two/model_best.pth  ./best_model/msr_action_two/model_best_2.pth

CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_3.list | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_3.log
cp output/msr_action_two/model_best.pth  ./best_model/msr_action_two/model_best_3.pth

CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_4.list | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_4.log
cp output/msr_action_two/model_best.pth  ./best_model/msr_action_two/model_best_4.pth

CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_5.list | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_5.log
cp output/msr_action_two/model_best.pth  ./best_model/msr_action_two/model_best_5.pth

# CUDA_VISIBLE_DEVICES=$GPUs python train-msr-twotrans.py --depth-xyz 4 --depth-t 3  --data-meta ./datasets/msr/msr_all.list  | tee output/msr_action_two/output_1GPU_xyz_4_t_3_msr_all.log