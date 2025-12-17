#!/usr/bin/env python3
"""
Batch process volleyball dataset images with D-FINE
Creates detection database for MOTRv2 training with class IDs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


class DetectionDatabase:
    """Create detection database in MOTRv2 format with class IDs"""

    def __init__(self, score_threshold=0.3, allowed_classes=None):
        self.score_threshold = score_threshold
        self.allowed_classes = allowed_classes or [0, 1]  # person, sports ball
        self.detection_db = {}
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'kept_detections': 0,
            'class_counts': {}
        }

    def process_frame(self, labels, boxes, scores, frame_key, width, height):
        """
        Process detections for a single frame

        Args:
            labels: Class IDs tensor
            boxes: Bounding boxes tensor [x1, y1, x2, y2]
            scores: Confidence scores tensor
            frame_key: Key for detection database (e.g., "volleyball_full/train/seq1/img1/000001")
            width: Image width
            height: Image height
        """
        detections = []

        for label, box, score in zip(labels, boxes, scores):
            label_id = label.item()
            conf = score.item()

            # Filter by score
            if conf < self.score_threshold:
                continue

            # Filter by allowed classes
            if label_id not in self.allowed_classes:
                continue

            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = box.tolist()
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            # MOTRv2 format (NO class ID): "x,y,w,h,conf\n"
            det_str = f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f}\n"
            detections.append(det_str)

            # Update stats
            self.stats['kept_detections'] += 1
            self.stats['class_counts'][label_id] = self.stats['class_counts'].get(label_id, 0) + 1

        self.detection_db[frame_key] = detections
        self.stats['total_images'] += 1

    def save(self, output_path):
        """Save detection database to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.detection_db, f, indent=2)

        print(f"\n{'='*80}")
        print("Detection Database Statistics:")
        print(f"{'='*80}")
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Total detections kept: {self.stats['kept_detections']}")
        print(f"Average detections per image: {self.stats['kept_detections']/max(1, self.stats['total_images']):.2f}")
        print(f"\nClass distribution:")
        for class_id, count in sorted(self.stats['class_counts'].items()):
            print(f"  Class {class_id}: {count} detections")
        print(f"{'='*80}\n")
        print(f"Detection database saved to: {output_path}")


def process_dataset(model, device, dataset_path, output_dir, split='train',
                   det_db=None, seq_name='seq1', start_frame=1):
    """
    Process all images in a dataset split

    Args:
        model: D-FINE model
        device: Torch device
        dataset_path: Path to dataset root (e.g., Datasets/Volleyball-Activity-Dataset-3)
        output_dir: Output directory for MOT format
        split: Dataset split ('train', 'valid', 'test')
        det_db: DetectionDatabase instance
        seq_name: Sequence name (default: 'seq1')
        start_frame: Starting frame number (default: 1)
    """
    # Input: COCO format images
    image_dir = Path(dataset_path) / split
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"\nProcessing {len(image_files)} images from {split} split...")

    # Prepare transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Process each image
    for idx, image_path in enumerate(tqdm(image_files, desc=f"Processing {split}")):
        # Load image
        im_pil = Image.open(image_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        # Run inference
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Create frame key for detection database
        # Format: "volleyball_full/train/seq1/img1/000001"
        frame_number = start_frame + idx
        frame_key = f"volleyball_full/{split}/{seq_name}/img1/{frame_number:06d}"

        # Process detections
        if det_db is not None:
            det_db.process_frame(
                labels[0], boxes[0], scores[0],
                frame_key, w, h
            )


def main(args):
    """Main function"""
    print(f"\n{'='*80}")
    print("D-FINE Batch Processing for MOTRv2 Training")
    print(f"{'='*80}\n")

    # Load D-FINE model
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("--resume required to load model checkpoint")

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

    device = torch.device(args.device)
    model = Model().to(device)
    model.eval()

    print(f"Model loaded: {args.resume}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_path}")

    # Parse allowed classes
    allowed_classes = [int(x.strip()) for x in args.allowed_classes.split(',')]
    print(f"Allowed classes: {allowed_classes}")
    print(f"Score threshold: {args.score_threshold}")

    # Initialize detection database
    det_db = DetectionDatabase(
        score_threshold=args.score_threshold,
        allowed_classes=allowed_classes
    )

    # Process each split
    for split in args.splits.split(','):
        split = split.strip()
        if split:
            process_dataset(
                model, device,
                args.dataset_path,
                args.output_dir,
                split=split,
                det_db=det_db,
                seq_name=args.seq_name,
                start_frame=args.start_frame
            )

    # Save detection database
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    det_db.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process volleyball dataset with D-FINE for MOTRv2 training"
    )

    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Path to D-FINE config file")
    parser.add_argument("-r", "--resume", type=str, required=True,
                       help="Path to D-FINE checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset (e.g., Datasets/Volleyball-Activity-Dataset-3)")
    parser.add_argument("--output-dir", type=str, default="MOTRv2/data/Dataset/mot",
                       help="Output directory for detection database")
    parser.add_argument("--output-filename", type=str, default="det_db_volleyball_dfine.json",
                       help="Output filename for detection database")
    parser.add_argument("--splits", type=str, default="train,valid",
                       help="Comma-separated list of splits to process")
    parser.add_argument("--seq-name", type=str, default="seq1",
                       help="Sequence name for MOT format")
    parser.add_argument("--start-frame", type=int, default=1,
                       help="Starting frame number")
    parser.add_argument("--allowed-classes", type=str, default="0,1,36,156,240",
                       help="Comma-separated list of class IDs to include")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                       help="Score threshold for detections")
    parser.add_argument("-d", "--device", type=str, default="cuda:0",
                       help="Device to use (e.g., cpu, cuda:0)")

    args = parser.parse_args()
    main(args)