#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0
#SBATCH --output=dfine_batch_volleyball_%j.out
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
cd /home/es/es_es/es_lekamt00/BeachKI

# Run D-FINE batch processing
python D-FINE/tools/inference/batch_process_dataset.py \
    -c D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml \
    -r D-FINE/dfine_x_obj365.pth \
    --dataset-path Datasets/Volleyball-Activity-Dataset-3 \
    --output-dir MOTRv2/data/Dataset/mot \
    --output-filename det_db_for_finetune_dfine.json \
    --splits train,valid \
    --allowed-classes "0,1,36,156,240" \
    --score-threshold 0.7 \
    -d cuda:0

echo ""
echo "========================================="
echo "Batch processing completed!"
echo "End time: $(date)"
echo "========================================="