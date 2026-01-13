#!/usr/bin/env python3
"""
Create D-FINE Detection Database for volleyball/finetune dataset

This script runs D-FINE on all images in the finetune dataset and creates
a detection database in MOTRv2 format for training.

Usage:
    cd D-FINE/tools/inference
    python create_detection_db_for_finetune.py
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Add D-FINE to path (from D-FINE/tools/inference to D-FINE root)
DFINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, DFINE_ROOT)
from src.core import YAMLConfig

# Import MOTRv2 formatter
sys.path.insert(0, os.path.join(DFINE_ROOT, "src/zoo/dfine"))
from motrv2_formatter import MOTRv2Formatter


def load_dfine_model(config_path, checkpoint_path, device='cuda'):
    """Load D-FINE model"""
    print(f"Loading D-FINE model...")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")

    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    print(f"Model loaded successfully on {device}")
    return model


def process_image_folder(model, device, image_folder, motrv2_formatter, sequence_name):
    """Process all images in a folder"""

    # Get all images
    image_files = sorted(list(Path(image_folder).glob("*.jpg")))
    print(f"\nFound {len(image_files)} images in {image_folder}")

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        # Extract frame number from filename (e.g., 000001.jpg -> 1)
        frame_number = int(img_file.stem)

        # Load image
        im_pil = Image.open(img_file).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        # Run D-FINE
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(im_data, orig_size)
            labels, boxes, scores = output

        # Add to MOTRv2 formatter
        motrv2_formatter.process_single_frame(
            labels[0], boxes[0], scores[0], sequence_name, frame_number, w, h
        )

    print(f"Processed {len(image_files)} images")


def main():
    # Get root directory (from D-FINE/tools/inference to BeachKI root)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

    # Configuration - all paths relative to BeachKI root
    DFINE_CONFIG = os.path.join(ROOT_DIR, "D-FINE/configs/dfine/objects365/dfine_hgnetv2_l_obj365.yml")
    DFINE_CHECKPOINT = os.path.join(ROOT_DIR, "D-FINE/dfine_l_obj365.pth")
    IMAGE_FOLDER = os.path.join(ROOT_DIR, "MOTRv2/data/Dataset/mot/volleyball/finetune/gt/img1")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "MOTRv2/data/Dataset/mot/det_db_volleyball_finetune_dfine.json")
    SEQUENCE_NAME = "volleyball/finetune/gt"

    # Settings from your command
    SCORE_THRESHOLD = 0.3
    ALLOWED_CLASSES = [0, 1, 36, 156, 240]  # From your command

    # Check if files exist
    if not os.path.exists(DFINE_CONFIG):
        print(f"ERROR: D-FINE config not found at {DFINE_CONFIG}")
        print("Please adjust the path in the script.")
        return

    if not os.path.exists(DFINE_CHECKPOINT):
        print(f"ERROR: D-FINE checkpoint not found at {DFINE_CHECKPOINT}")
        print("Please adjust the path in the script.")
        return

    if not os.path.exists(IMAGE_FOLDER):
        print(f"ERROR: Image folder not found at {IMAGE_FOLDER}")
        return

    print("="*80)
    print("Creating D-FINE Detection Database for MOTRv2 Training")
    print("="*80)
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"Sequence name: {SEQUENCE_NAME}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"Allowed classes: {ALLOWED_CLASSES}")
    print(f"Output: {OUTPUT_PATH}")
    print("="*80)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load model
    model = load_dfine_model(DFINE_CONFIG, DFINE_CHECKPOINT, device)

    # Initialize MOTRv2 formatter
    motrv2_formatter = MOTRv2Formatter(
        score_threshold=SCORE_THRESHOLD,
        allowed_classes=ALLOWED_CLASSES
    )

    # Process all images
    process_image_folder(model, device, IMAGE_FOLDER, motrv2_formatter, SEQUENCE_NAME)

    # Save detection database
    print(f"\nSaving detection database to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(motrv2_formatter.detection_database, f)

    # Print statistics
    print("\n" + "="*80)
    print("Statistics:")
    print("="*80)
    stats = motrv2_formatter.stats
    print(f"Total detections processed: {stats['total_detections']}")
    print(f"Filtered by score: {stats['filtered_by_score']}")
    print(f"Filtered by class: {stats['filtered_by_class']}")
    print(f"Kept detections: {stats['kept_detections']}")
    print(f"\nClass distribution:")
    for class_id, count in sorted(stats['class_counts'].items()):
        print(f"  Class {class_id}: {count} detections")
    print("="*80)
    print(f"\nDetection database created successfully!")
    print(f"Location: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"1. Update your training config to use: det_db_volleyball_finetune_dfine.json")
    print(f"2. Re-run MOTRv2 training with the new detection database")
    print("="*80)


if __name__ == "__main__":
    main()
