#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0
#SBATCH --output=dfine_gt_%j.out
#SBATCH --job-name=dfine_vb_batch

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch

# Print job info
echo "========================================="
echo "D-FINE Batch Processing (Volleyball)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Go to BeachKI directory
cd /home/es/es_es/es_lekamt00/BeachKI/D-FINE

# Run D-FINE batch processing
python tools/inference/torch_inf.py \
    -c configs/dfine/objects365/dfine_hgnetv2_l_obj365.yml \
    -r dfine_l_obj365.pth \
    -i "/home/es/es_es/es_lekamt00/BeachKI/Datasets/Volleyball-Activity-Dataset-3/combined/*.jpg" \
    -d cuda:0 \
    --motrv2 \
    --sequence-name "volleyball/combined/gt" \
    --allowed-classes "0,1,36,156,240" \
    --motrv2-score-threshold 0.5

echo ""
echo "========================================="
echo "Batch processing completed!"
echo "End time: $(date)"
echo "========================================="