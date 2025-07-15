"""
Interactive Video Processor for D-FINE to MOTRv2
-------------------------------------------------
This script allows for:
1. Interactive selection of a court area on the first frame of a video.
2. Saving/loading the court configuration for reuse.
3. Processing a video with D-FINE.
4. Applying both Object ID translation and court area filtering.
5. Saving the final, clean detections to a MOTRv2-compatible JSON file.
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
from src.zoo.dfine.motrv2_formatter import CourtAreaSelector, EnhancedMOTRv2Formatter


def get_first_frame(video_path, output_path="first_frame.jpg"):
    """Extracts the first frame of a video and saves it as an image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read the first frame from {video_path}")
        
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"First frame saved to {output_path}")
    return output_path


def main(args):
    # --- 1. Setup Model ---
    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
    cfg.model.load_state_dict(state)
    
    model = torch.nn.Sequential(cfg.model, cfg.postprocessor).to(args.device)
    model.eval()
    
    # --- 2. Interactive Court Selection ---
    court_config_path = "court_config.json"
    court_bounds = None
    
    if os.path.exists(court_config_path):
        choice = input(f"Found existing court config at '{court_config_path}'. Use it? (y/n): ").lower()
        if choice == 'y':
            court_bounds = CourtAreaSelector.load_court_config(court_config_path)
            print("Loaded existing court configuration.")
    
    if court_bounds is None:
        print("Starting interactive court selection...")
        first_frame_path = get_first_frame(args.input_video)
        selector = CourtAreaSelector()
        court_bounds = selector.select_court_area(first_frame_path)
        selector.save_court_config(court_config_path)

    # --- 3. Process Video ---
    formatter = EnhancedMOTRv2Formatter(
        score_threshold=args.score_threshold,
        court_bounds=court_bounds,
        enable_translation=True
    )

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.input_video}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Prepare frame for D-FINE
        h, w = frame.shape[:2]
        orig_size = torch.tensor([[w, h]]).to(args.device)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        img_data = transforms(img_pil).unsqueeze(0).to(args.device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_data, orig_size)
            labels, boxes, scores = outputs[0]

        # Process and filter detections
        formatter.process_single_frame(
            labels=labels,
            boxes=boxes,
            scores=scores,
            sequence_name=args.sequence_name,
            frame_number=frame_count,
            image_width=w,
            image_height=h
        )

    cap.release()
    print("\nVideo processing complete.")

    # --- 4. Save Results ---
    formatter.save_database(output_path=os.path.dirname(args.input_video) or '.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive D-FINE Video Processor for MOTRv2")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to model config file")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument('--score_threshold', type=float, default=0.4, help="Confidence score threshold for detections")
    parser.add_argument('--sequence_name', type=str, default='beach_volleyball_seq', help="Name for this video sequence in the JSON output")
    args = parser.parse_args()
    main(args)