"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified to support MOTRv2 output format
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

# Import MOTRv2 formatter
try:
    from src.zoo.dfine.motrv2_formatter import MOTRv2Formatter, MOTRv2VideoProcessor
    MOTRV2_AVAILABLE = True
except ImportError:
    MOTRV2_AVAILABLE = False
    print("Warning: MOTRv2 formatter not found. MOTRv2 output will be disabled.")


def draw(images, labels, boxes, scores, thrh=0.4):
    """Original drawing function for visualization"""
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

        im.save("torch_results.jpg")


def process_image(model, device, file_path, motrv2_formatter=None, sequence_name=None):
    """
    Process a single image with optional MOTRv2 output
    
    Args:
        model: D-FINE model
        device: Torch device
        file_path: Path to image
        motrv2_formatter: Optional MOTRv2Formatter instance
        sequence_name: Optional sequence name for MOTRv2 output
    """
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # Original visualization
    draw([im_pil], labels, boxes, scores)
    
    # MOTRv2 formatting if requested
    if motrv2_formatter is not None:
        if sequence_name is None:
            sequence_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process single frame (frame number 1 for single images)
        motrv2_formatter.process_single_frame(
            labels[0], boxes[0], scores[0], sequence_name, 1
        )
        
        # Save MOTRv2 JSON
        output_dir = os.path.dirname(file_path) or "."
        motrv2_formatter.save_database(output_dir)
        print(f"MOTRv2 format saved for image: {file_path}")


def process_video(model, device, file_path, motrv2_formatter=None, sequence_name=None):
    """
    Process a video with optional MOTRv2 output
    
    Args:
        model: D-FINE model
        device: Torch device
        file_path: Path to video
        motrv2_formatter: Optional MOTRv2Formatter instance
        sequence_name: Optional sequence name for MOTRv2 output
    """
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    # Set default sequence name if not provided
    if sequence_name is None:
        sequence_name = os.path.splitext(os.path.basename(file_path))[0]

    frame_count = 0
    print("Processing video frames...")
    
    # For MOTRv2, we'll collect all detections if formatter is provided
    if motrv2_formatter is not None:
        print("MOTRv2 formatting enabled - will save detections after processing")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Process MOTRv2 format if enabled
        if motrv2_formatter is not None:
            # Frame numbers are 1-based for MOTRv2
            motrv2_formatter.process_single_frame(
                labels[0], boxes[0], scores[0], sequence_name, frame_count + 1
            )

        # Draw detections on the frame for visualization
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    
    print(f"Video processing complete. Processed {frame_count} frames.")
    print("Result saved as 'torch_results.mp4'.")
    
    # Save MOTRv2 detections if formatter was provided
    if motrv2_formatter is not None:
        output_dir = os.path.dirname(file_path) or "."
        json_path = motrv2_formatter.save_database(output_dir)
        print(f"MOTRv2 detections saved to: {json_path}")


def main(args):
    """Main function with MOTRv2 support"""
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
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
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

    device = args.device
    model = Model().to(device)

    # Initialize MOTRv2 formatter if requested
    motrv2_formatter = None
    if args.motrv2 and MOTRV2_AVAILABLE:
        print("MOTRv2 output format enabled")
        
        # Configure MOTRv2 formatter
        category_mapping = None
        if args.motrv2_category_mapping:
            # Parse category mapping from command line
            # Format: "0:1,2:1,3:2" maps D-FINE classes 0,2 to MOTRv2 class 1, and 3 to 2
            try:
                mapping_pairs = args.motrv2_category_mapping.split(',')
                category_mapping = {}
                for pair in mapping_pairs:
                    dfine_id, motrv2_id = map(int, pair.split(':'))
                    category_mapping[dfine_id] = motrv2_id
                print(f"Using category mapping: {category_mapping}")
            except:
                print("Warning: Invalid category mapping format. Using identity mapping.")
        
        motrv2_formatter = MOTRv2Formatter(
            score_threshold=args.motrv2_score_threshold,
            category_mapping=category_mapping
        )
    elif args.motrv2 and not MOTRV2_AVAILABLE:
        print("Error: MOTRv2 formatter not available. Please ensure motrv2_formatter.py is in the correct location.")
        return

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path, motrv2_formatter, args.sequence_name)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path, motrv2_formatter, args.sequence_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, 
                        help="Path to config file")
    parser.add_argument("-r", "--resume", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input image or video")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="Device to use (e.g., cpu, cuda:0)")
    
    # MOTRv2 specific arguments
    parser.add_argument("--motrv2", action="store_true",
                        help="Enable MOTRv2 output format")
    parser.add_argument("--motrv2-score-threshold", type=float, default=0.3,
                        help="Score threshold for MOTRv2 detections (default: 0.3)")
    parser.add_argument("--sequence-name", type=str, default=None,
                        help="Sequence name for MOTRv2 output (default: input filename)")
    parser.add_argument("--motrv2-category-mapping", type=str, default=None,
                        help="Category mapping for MOTRv2 format, e.g., '0:1,2:1,3:2'")
    
    args = parser.parse_args()
    main(args)