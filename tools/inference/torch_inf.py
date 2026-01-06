"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified to support simplified MOTRv2 output format
"""

import os
import sys
import json
import glob

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


def process_image(model, device, file_path, motrv2_formatter=None, sequence_name=None, frame_number=1):
    """
    Process a single image with optional MOTRv2 output

    Args:
        model: D-FINE model
        device: Torch device
        file_path: Path to image
        motrv2_formatter: Optional MOTRv2Formatter instance
        sequence_name: Optional sequence name for MOTRv2 output
        frame_number: Frame number for MOTRv2 output (default: 1)
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

    # Original visualization (only for single image processing)
    if frame_number == 1:
        draw([im_pil], labels, boxes, scores)

    # MOTRv2 formatting if requested
    if motrv2_formatter is not None:
        if sequence_name is None:
            sequence_name = os.path.splitext(os.path.basename(file_path))[0]

        # Process frame with specified frame number
        motrv2_formatter.process_single_frame(
            labels[0], boxes[0], scores[0], sequence_name, frame_number, w, h
        )


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
                labels[0], boxes[0], scores[0], sequence_name, frame_count + 1, w, h
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
    """Main function with simplified MOTRv2 support"""
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
        
        # Parse allowed classes from command line or config file
        allowed_classes = None
        if args.motrv2_config:
            # Load from config file
            try:
                with open(args.motrv2_config, 'r') as f:
                    config = json.load(f)
                    allowed_classes = config.get('allowed_classes', [0, 1])
                    print(f"Loaded allowed classes from config: {allowed_classes}")
            except Exception as e:
                print(f"Error loading config file: {e}. Using command line arguments.")
                allowed_classes = None
        
        if allowed_classes is None:
            # Parse from command line
            try:
                allowed_classes = [int(x.strip()) for x in args.allowed_classes.split(',')]
                print(f"Using allowed classes from command line: {allowed_classes}")
            except:
                print("Error parsing allowed classes. Using default [0, 1]")
                allowed_classes = [0, 1]
        
        motrv2_formatter = MOTRv2Formatter(
            score_threshold=args.motrv2_score_threshold,
            allowed_classes=allowed_classes
        )
    elif args.motrv2 and not MOTRV2_AVAILABLE:
        print("Error: MOTRv2 formatter not available. Please ensure motrv2_formatter.py is in the correct location.")
        return

    # Check if input contains glob patterns or is a single file
    input_path = args.input

    # Expand glob patterns
    if '*' in input_path or '?' in input_path:
        image_files = sorted(glob.glob(input_path))
        if not image_files:
            print(f"Error: No files found matching pattern: {input_path}")
            return
        print(f"Found {len(image_files)} images matching pattern")
    else:
        image_files = [input_path]

    # Determine if we're processing images or video
    first_file = image_files[0]
    file_ext = os.path.splitext(first_file)[-1].lower()

    if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image(s)
        if len(image_files) == 1:
            # Single image
            process_image(model, device, image_files[0], motrv2_formatter, args.sequence_name)
            if motrv2_formatter is not None:
                output_dir = os.path.dirname(image_files[0]) or "."
                motrv2_formatter.save_database(output_dir)
            print("Image processing complete.")
        else:
            # Multiple images - process as sequence
            print(f"Processing {len(image_files)} images as sequence...")
            for idx, image_file in enumerate(image_files, start=1):
                process_image(
                    model, device, image_file, motrv2_formatter,
                    args.sequence_name, frame_number=idx
                )
                if idx % 10 == 0:
                    print(f"Processed {idx}/{len(image_files)} images...")

            # Save MOTRv2 database after all images are processed
            if motrv2_formatter is not None:
                output_dir = os.path.dirname(image_files[0]) or "."
                json_path = motrv2_formatter.save_database(output_dir)
                print(f"MOTRv2 detections saved to: {json_path}")

            print(f"Image sequence processing complete. Processed {len(image_files)} images.")
    else:
        # Process as video
        if len(image_files) > 1:
            print("Warning: Multiple video files detected. Only processing the first one.")
        process_video(model, device, image_files[0], motrv2_formatter, args.sequence_name)


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
    parser.add_argument("--allowed-classes", type=str, default="0,1",
                        help="Comma-separated list of class IDs to include (default: '0,1')")
    parser.add_argument("--motrv2-config", type=str, default=None,
                        help="Path to MOTRv2 config JSON file")
    
    args = parser.parse_args()
    main(args)