#!/bin/bash
#$ -cwd
#$ -V
#$ -N frozen_lgbm
#$ -o logs/frozen_lgbm.$JOB_ID.out
#$ -e logs/frozen_lgbm.$JOB_ID.err
#$ -l gpu,A6000,h_rt=8:00:00,h_data=16G
#$ -pe shared 4

echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "======================================"

mkdir -p logs

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate antibiotics

cd /u/project/cluo/emmatu/projects/antibiotics-fm-benchmark

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================"
echo "CUDA_VISIBLE_DEVICES before run: $CUDA_VISIBLE_DEVICES"
echo "GPU status before run:"
nvidia-smi
echo "======================================"

echo "Python/PyTorch check:"
python -c "import torch, transformers; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('transformers:', transformers.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "======================================"
echo "Starting frozen BioClinicalBERT + LightGBM pipeline..."
echo "======================================"

python run_frozen_validation_with_predictions.py

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"