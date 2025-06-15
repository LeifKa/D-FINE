"""
Batch processing script for generating MOTRv2-compatible detection files from multiple video sequences.
This script is designed for processing entire datasets for MOTRv2 tracking evaluation.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.zoo.dfine.motrv2_formatter import MOTRv2Formatter, MOTRv2VideoProcessor


class DFINEBatchProcessor:
    """
    Batch processor for converting D-FINE detections to MOTRv2 format.
    Handles multiple video sequences and generates a single detection database.
    """
    
    def __init__(self, model, device, transform, score_threshold=0.3):
        """
        Initialize the batch processor.
        
        Args:
            model: D-FINE model instance
            device: Torch device
            transform: Image transformation pipeline
            score_threshold: Minimum confidence score for detections
        """
        self.model = model
        self.device = device
        self.transform = transform
        self.formatter = MOTRv2Formatter(score_threshold=score_threshold)
        
    def process_video_sequence(self, video_path, sequence_name=None, max_frames=None):
        """
        Process a single video sequence.
        
        Args:
            video_path: Path to video file
            sequence_name: Name for the sequence (default: video filename)
            max_frames: Maximum number of frames to process (None for all)
        """
        if sequence_name is None:
            sequence_name = Path(video_path).stem
            
        print(f"\nProcessing sequence: {sequence_name}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        frame_count = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {sequence_name}")
        
        while cap.isOpened() and (max_frames is None or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(self.device)
            
            # Transform and run inference
            im_data = self.transform(frame_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                labels, boxes, scores = self.model(im_data, orig_size)
            
            # Process detections for MOTRv2
            self.formatter.process_single_frame(
                labels[0], boxes[0], scores[0], 
                sequence_name, frame_count + 1  # 1-based frame indexing
            )
            
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        print(f"Processed {frame_count} frames from {sequence_name}")
        
    def process_image_sequence(self, image_folder, sequence_name=None, pattern="*.jpg"):
        """
        Process a sequence of images from a folder.
        
        Args:
            image_folder: Path to folder containing images
            sequence_name: Name for the sequence (default: folder name)
            pattern: Glob pattern for image files
        """
        if sequence_name is None:
            sequence_name = Path(image_folder).name
            
        print(f"\nProcessing image sequence: {sequence_name}")
        
        # Get sorted list of images
        image_files = sorted(glob.glob(os.path.join(image_folder, pattern)))
        
        if not image_files:
            print(f"No images found in {image_folder} with pattern {pattern}")
            return
            
        for frame_idx, image_path in enumerate(tqdm(image_files, desc=f"Processing {sequence_name}")):
            # Load image
            im_pil = Image.open(image_path).convert("RGB")
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]]).to(self.device)
            
            # Transform and run inference
            im_data = self.transform(im_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                labels, boxes, scores = self.model(im_data, orig_size)
            
            # Process detections for MOTRv2
            self.formatter.process_single_frame(
                labels[0], boxes[0], scores[0], 
                sequence_name, frame_idx + 1  # 1-based frame indexing
            )
            
        print(f"Processed {len(image_files)} images from {sequence_name}")
        
    def save_detections(self, output_path, filename="det_db_motrv2.json"):
        """Save all processed detections to JSON file."""
        return self.formatter.save_database(output_path, filename)


def load_dfine_model(config_path, checkpoint_path, device):
    """Load D-FINE model from config and checkpoint."""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
        
    cfg.model.load_state_dict(state)
    
    # Create deployable model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
            
    return Model().to(device)


def main():
    parser = argparse.ArgumentParser(description="Batch process videos/images for MOTRv2 with D-FINE")
    
    # Model configuration
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to D-FINE config file")
    parser.add_argument("-r", "--checkpoint", type=str, required=True,
                        help="Path to D-FINE checkpoint")
    parser.add_argument("-d", "--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    
    # Input/output configuration
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory containing videos or image folders")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for MOTRv2 JSON file")
    parser.add_argument("--input-type", type=str, choices=["video", "images"], default="video",
                        help="Type of input: 'video' for video files, 'images' for image sequences")
    
    # Processing options
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Score threshold for detections (default: 0.3)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process per video (default: all)")
    parser.add_argument("--image-pattern", type=str, default="*.jpg",
                        help="Pattern for image files when using image input (default: *.jpg)")
    parser.add_argument("--video-pattern", type=str, default="*.mp4",
                        help="Pattern for video files (default: *.mp4)")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading D-FINE model...")
    model = load_dfine_model(args.config, args.checkpoint, args.device)
    model.eval()
    
    # Setup image transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Initialize batch processor
    processor = DFINEBatchProcessor(
        model=model,
        device=args.device,
        transform=transforms,
        score_threshold=args.score_threshold
    )
    
    # Process based on input type
    if args.input_type == "video":
        # Find all video files
        video_files = glob.glob(os.path.join(args.input, args.video_pattern))
        if not video_files:
            video_files = glob.glob(os.path.join(args.input, "**", args.video_pattern), recursive=True)
            
        print(f"Found {len(video_files)} video files")
        
        for video_path in video_files:
            processor.process_video_sequence(video_path, max_frames=args.max_frames)
            
    else:  # images
        # Find all subdirectories (each is a sequence)
        sequence_dirs = [d for d in Path(args.input).iterdir() if d.is_dir()]
        
        if not sequence_dirs:
            # Treat input directory as a single sequence
            sequence_dirs = [Path(args.input)]
            
        print(f"Found {len(sequence_dirs)} image sequences")
        
        for seq_dir in sequence_dirs:
            processor.process_image_sequence(str(seq_dir), pattern=args.image_pattern)
    
    # Save results
    print("\nSaving MOTRv2 detection database...")
    output_path = processor.save_detections(args.output)
    print(f"\nProcessing complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()