"""
Manual Coordinate Video Processor for D-FINE to MOTRv2 (No GUI)
--- FINAL CORRECTED VERSION (using .deploy()) ---
"""

import os
import sys
import argparse
import torch
import cv2
from PIL import Image
import torchvision.transforms as T

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core import YAMLConfig
from src.zoo.dfine.motrv2_formatter import EnhancedMOTRv2Formatter


def get_first_frame(video_path, output_path="first_frame.jpg"):
    """Extracts the first frame of a video and saves it as an image."""
    print(f"Extracting first frame from '{video_path}'...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read the first frame from {video_path}")
        
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"Success! First frame saved to '{output_path}'.")
    print("Please download this file to your local PC to determine court coordinates.")


def main(args):
    # --- Special Mode: Just get the first frame and exit ---
    if args.get_first_frame:
        get_first_frame(args.input_video)
        sys.exit(0)

    # --- 1. Validate input ---
    if not args.court_bounds:
        print("Error: --court-bounds is required for processing.")
        print("Please run with --get-first-frame first to determine the coordinates.")
        sys.exit(1)

    try:
        coords = [int(c.strip()) for c in args.court_bounds.split(',')]
        if len(coords) != 4: raise ValueError
        x1, y1, x2, y2 = coords
    except (ValueError, TypeError):
        print("Error: --court-bounds format must be four integers separated by commas, e.g., '100,150,800,750'")
        sys.exit(1)

    # --- 2. Setup Model (THE CORRECT WAY) ---
    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    
    checkpoint = torch.load(args.resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    
    cfg.model.load_state_dict(state)
    
    # Use .deploy() to get clean, inference-ready model components
    d_fine_model = cfg.model.deploy().to(args.device)
    postprocessor = cfg.postprocessor.deploy().to(args.device)
    d_fine_model.eval()

    # --- 3. Calculate Court Ratios for Formatter ---
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.input_video}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    court_bounds_ratios = {
        'left': min(x1, x2) / video_width,
        'right': max(x1, x2) / video_width,
        'top': min(y1, y2) / video_height,
        'bottom': max(y1, y2) / video_height,
    }
    print(f"Using court bounds (pixels): Top-Left({min(x1,x2)}, {min(y1,y2)}), Bottom-Right({max(x1,x2)}, {max(y1,y2)})")

    # --- 4. Process Video ---
    formatter = EnhancedMOTRv2Formatter(
        score_threshold=args.score_threshold,
        court_bounds_ratios=court_bounds_ratios,
        enable_translation=True
    )

    cap = cv2.VideoCapture(args.input_video)
    frame_count = 0
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        h, w = frame.shape[:2]
        orig_size = torch.tensor([[w, h]], device=args.device)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        img_data = transforms(img_pil).unsqueeze(0).to(args.device)

        with torch.no_grad():
            raw_outputs = d_fine_model(img_data)
            # The postprocessor returns a list (for batches), so we get the first element [0]
            proc_labels, proc_boxes, proc_scores = postprocessor(raw_outputs, orig_size)[0]

        formatter.process_single_frame(
            in_labels=proc_labels,
            in_boxes=proc_boxes,
            in_scores=proc_scores,
            sequence_name=args.sequence_name,
            frame_number=frame_count,
            image_width=w,
            image_height=h
        )

    cap.release()
    print("\nVideo processing complete.")
    formatter.save_database_to_json(output_dir=os.path.dirname(args.input_video) or '.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manual Coordinate Video Processor for D-FINE (No GUI)")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to model config file")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use")
    parser.add_argument('--score_threshold', type=float, default=0.4, help="Confidence score threshold")
    parser.add_argument('--sequence_name', type=str, default='beach_volleyball_seq', help="Name for this video sequence")
    parser.add_argument('--get-first-frame', action='store_true', help="If set, only extracts the first frame and exits.")
    parser.add_argument('--court-bounds', type=str, help='Bounding box of the court in pixels, format: "x1,y1,x2,y2"')
    args = parser.parse_args()
    main(args)