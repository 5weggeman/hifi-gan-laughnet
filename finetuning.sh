#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ft

source /scratch/$USER/.envs/thesis_env/bin/activate

module purge
module load Pillow/9.1.0-GCCcore-10.3.0 
module load Python/3.9.5-GCCcore-10.3.0 
module load SoX/14.4.2-GCCcore-10.3.0
module load CUDA/10.2.89

pip install -r /scratch/s5007453/hifi-gan-laughnet/requirements.txt

python train.py --fine_tuning=True --input_wavs_dir='/scratch/s5007453/hifi-gan-laughnet/laughter/output' --input_training_file='/scratch/s5007453/hifi-gan-laughnet/laughter/output/training-ft.txt' --input_validation_file='/scratch/s5007453/hifi-gan-laughnet/laughter/output/validation-ft.txt' --checkpoint_path='/scratch/s5007453/hifi-gan-laughnet/cp_hifigan' --config='config_v1.json' --checkpoint_interval='5000' --training_epochs='50055'