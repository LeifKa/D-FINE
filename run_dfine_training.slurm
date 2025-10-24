#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:0:0
#SBATCH --nodes=1
#SBATCH --output=dfine_training_%j.out
#SBATCH --job-name=dfine_train_det

# Load modules
module load devel/miniforge/24.11.0-py3.12
module load devel/cuda/12.4

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Navigate to D-FINE directory
cd /home/es/es_es/es_lekamt00/BeachKI/D-FINE

# Create training config for D-FINE (lower threshold for more detections)
cat > motrv2_training_config.json << 'EOF'
{
  "allowed_classes": [0],
  "score_threshold": 0.3
}
EOF

echo "Generating D-FINE detections on training data..."
echo "Using config: motrv2_training_config.json"
echo "Video: Sequenz_Beach.mp4"

# Run D-FINE on training video
python tools/inference/torch_inf.py \
    -c configs/dfine/objects365/dfine_hgnetv2_l_obj365.yml \
    -r dfine_hgnetv2_l_obj365.pth \
    --input Sequenz_Beach.mp4 \
    --motrv2 \
    --motrv2-config motrv2_training_config.json \
    --sequence-name volleyball/train/game1

echo "D-FINE detection complete!"
echo "Results saved to: det_db_motrv2.json"

# Copy detections to MOTRv2 directory
cp det_db_motrv2.json ../MOTRv2/det_db_volleyball_train_dfine.json
echo "Copied to: MOTRv2/det_db_volleyball_train_dfine.json"
