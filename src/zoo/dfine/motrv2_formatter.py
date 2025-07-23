"""
Enhanced MOTRv2 Formatter - Fixed for correct MOTRv2 JSON format
Compatible with MOTRv2's expected input format
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CourtAreaSelector:
    """Interactive tool for manually selecting the court area in a video frame."""
    
    def __init__(self):
        self.points = []
        self.court_bounds = None
        
    def select_court_area(self, image_path: str) -> Dict[str, float]:
        """Interactive method to select court boundaries by clicking on the image."""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.set_title("Click 4 corners of the court area (or press 'c' to use center region)")
        
        self.points = []
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                if len(self.points) < 4:
                    self.points.append((event.xdata, event.ydata))
                    ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
                    
                    if len(self.points) > 1:
                        ax.plot([self.points[-2][0], self.points[-1][0]], 
                               [self.points[-2][1], self.points[-1][1]], 'r-', linewidth=2)
                    
                    if len(self.points) == 4:
                        ax.plot([self.points[-1][0], self.points[0][0]], 
                               [self.points[-1][1], self.points[0][1]], 'r-', linewidth=2)
                        ax.set_title("Court area selected! Press 'q' to confirm or 'r' to reset")
                    
                    plt.draw()
        
        def onkey(event):
            if event.key == 'r':
                self.points = []
                ax.clear()
                ax.imshow(img_rgb)
                ax.set_title("Click 4 corners of the court area (or press 'c' to use center region)")
                plt.draw()
            elif event.key == 'c':
                self.points = [
                    (width * 0.1, height * 0.1),
                    (width * 0.9, height * 0.1),
                    (width * 0.9, height * 0.9),
                    (width * 0.1, height * 0.9)
                ]
                ax.clear()
                ax.imshow(img_rgb)
                rect = patches.Rectangle((self.points[0][0], self.points[0][1]),
                                       width * 0.8, height * 0.8,
                                       linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.set_title("Using center region. Press 'q' to confirm")
                plt.draw()
            elif event.key == 'q':
                plt.close()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()
        
        if len(self.points) >= 4:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            
            self.court_bounds = {
                'left': min(xs) / width,
                'right': max(xs) / width,
                'top': min(ys) / height,
                'bottom': max(ys) / height
            }
            
            print(f"Court boundaries selected:")
            print(f"  Left: {self.court_bounds['left']:.2%} of image width")
            print(f"  Right: {self.court_bounds['right']:.2%} of image width") 
            print(f"  Top: {self.court_bounds['top']:.2%} of image height")
            print(f"  Bottom: {self.court_bounds['bottom']:.2%} of image height")
            
            return self.court_bounds
        else:
            print("No court area selected. Using full image.")
            return {'left': 0, 'right': 1, 'top': 0, 'bottom': 1}
    
    def save_court_config(self, config_path: str = "court_config.json"):
        """Save court configuration to file for reuse."""
        if self.court_bounds:
            with open(config_path, 'w') as f:
                json.dump(self.court_bounds, f, indent=2)
            print(f"Court configuration saved to {config_path}")
    
    @staticmethod
    def load_court_config(config_path: str = "court_config.json") -> Dict[str, float]:
        """Load court configuration from file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"No court configuration found at {config_path}")
            return None


class MOTRv2Formatter:
    """
    MOTRv2 Formatter - Creates the exact JSON format expected by MOTRv2
    
    MOTRv2 expects:
    {
        "sequence/img1/000001.txt": ["x,y,w,h,score\n", "x,y,w,h,score\n", ...],
        "sequence/img1/000002.txt": ["x,y,w,h,score\n", ...],
        ...
    }
    """
    
    # Beach volleyball specific mappings
    OBJECTS365_TO_COCO = {
        0: 0,      # Person -> person (players)
        1: 1,
        240: 37,   # Volleyball/Ball -> sports ball
    }
    
    CLASS_NAMES = {
        0: "player",
        37: "ball"
    }
    
    def __init__(self, 
                 score_threshold: float = 0.3,
                 court_bounds: Optional[Dict[str, float]] = None,
                 enable_translation: bool = True,
                 allowed_classes: Optional[List[int]] = None,
                 sequence_base_path: str = "img1",  # MOTRv2 expects "img1" folder
                 category_mapping: Optional[Dict[int, int]] = None):
        """
        Initialize the MOTRv2 formatter.
        
        Args:
            score_threshold: Minimum confidence score
            court_bounds: Court area boundaries  
            enable_translation: Whether to translate class IDs
            allowed_classes: List of allowed COCO class IDs
            sequence_base_path: Base path for sequence (usually "img1")
            category_mapping: Legacy parameter (ignored)
        """
        self.score_threshold = score_threshold
        self.court_bounds = court_bounds
        self.enable_translation = enable_translation
        self.sequence_base_path = sequence_base_path
        
        # For beach volleyball, default to only players and ball
        self.allowed_classes = allowed_classes or [0, 37]  # person, sports ball
        
        # MOTRv2 format: {"path/to/frame.txt": ["x,y,w,h,score\n", ...]}
        self.detection_database = {}
        
        self.stats = {
            'total_detections': 0,
            'filtered_by_score': 0,
            'filtered_by_court': 0,
            'filtered_by_translation': 0,
            'filtered_by_class': 0,
            'kept_detections': 0,
            'class_counts': {class_id: 0 for class_id in self.allowed_classes}
        }
        
        print(f"MOTRv2 formatter initialized:")
        print(f"  Allowed classes: {[self.CLASS_NAMES.get(cid, f'class_{cid}') for cid in self.allowed_classes]}")
        print(f"  Score threshold: {score_threshold}")
        print(f"  Court area filtering: {'enabled' if court_bounds else 'disabled'}")
        print(f"  Sequence path format: {sequence_base_path}/XXXXXX.txt")
    
    def translate_class_id(self, objects365_id: int) -> Optional[int]:
        """Translate Objects365 class ID to COCO class ID."""
        if not self.enable_translation:
            return objects365_id if objects365_id in self.allowed_classes else None
            
        coco_id = self.OBJECTS365_TO_COCO.get(objects365_id)
        
        if coco_id is None:
            return None
            
        if coco_id not in self.allowed_classes:
            return None
            
        return coco_id
    
    def is_in_court_area(self, bbox: List[float], image_width: int, image_height: int) -> bool:
        """Check if detection center is within court boundaries."""
        if self.court_bounds is None:
            return True
        
        # Calculate center of bounding box
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        # Convert court bounds from ratios to pixels
        court_left = self.court_bounds['left'] * image_width
        court_right = self.court_bounds['right'] * image_width
        court_top = self.court_bounds['top'] * image_height
        court_bottom = self.court_bounds['bottom'] * image_height
        
        return (court_left <= center_x <= court_right and 
                court_top <= center_y <= court_bottom)
    
    def convert_bbox_format(self, bbox: torch.Tensor) -> List[float]:
        """
        Convert from D-FINE format [x1, y1, x2, y2] to MOTRv2 format [x, y, width, height].
        
        MOTRv2 expects: x,y = top-left corner, width,height = dimensions
        """
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
            
        x1, y1, x2, y2 = bbox
        
        # Convert to top-left corner + dimensions format
        x = min(x1, x2)  # Left edge
        y = min(y1, y2)  # Top edge 
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        return [float(x), float(y), float(width), float(height)]
    
    def get_frame_path(self, sequence_name: str, frame_number: int) -> str:
        """
        Generate the frame path expected by MOTRv2.
        
        For sequence "Testsequenz1" and frame 1, returns: "Testsequenz1/img1/000001.txt"
        """
        frame_str = f"{frame_number:06d}"  # Zero-padded to 6 digits
        return f"{sequence_name}/{self.sequence_base_path}/{frame_str}.txt"
    
    def process_single_frame(self, 
                           labels: torch.Tensor,
                           boxes: torch.Tensor, 
                           scores: torch.Tensor,
                           sequence_name: str,
                           frame_number: int,
                           image_width: int = 640,
                           image_height: int = 640) -> Dict[str, List[str]]:
        """
        Process detections and format for MOTRv2.
        
        Returns:
            Dict with frame path as key and list of detection strings as value
        """
        # Convert tensors to numpy
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        frame_detections = []
        frame_stats = {
            'total': len(labels),
            'score_filtered': 0,
            'translation_filtered': 0,
            'court_filtered': 0,
            'kept': 0,
            'class_counts': {class_id: 0 for class_id in self.allowed_classes}
        }
        
        for i in range(len(labels)):
            self.stats['total_detections'] += 1
            
            # Stage 1: Score threshold
            if scores[i] < self.score_threshold:
                self.stats['filtered_by_score'] += 1
                frame_stats['score_filtered'] += 1
                continue
            
            # Stage 2: Class ID translation and filtering
            objects365_id = int(labels[i])
            coco_id = self.translate_class_id(objects365_id)
            
            if coco_id is None:
                self.stats['filtered_by_translation'] += 1
                frame_stats['translation_filtered'] += 1
                continue
            
            # Stage 3: Convert bbox format (D-FINE [x1,y1,x2,y2] -> MOTRv2 [x,y,w,h])
            bbox = self.convert_bbox_format(boxes[i])
            
            # Stage 4: Court area filtering
            if not self.is_in_court_area(bbox, image_width, image_height):
                self.stats['filtered_by_court'] += 1
                frame_stats['court_filtered'] += 1
                continue
            
            # Detection passed all filters - format for MOTRv2
            # MOTRv2 expects: "x,y,w,h,score\n"
            detection_string = f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},{scores[i]:.6f}\n"
            frame_detections.append(detection_string)
            
            self.stats['kept_detections'] += 1
            self.stats['class_counts'][coco_id] += 1
            frame_stats['kept'] += 1
            frame_stats['class_counts'][coco_id] += 1
        
        # Enhanced logging
        if frame_number % 30 == 0:
            class_summary = ", ".join([
                f"{self.CLASS_NAMES.get(cid, f'class_{cid}')}: {frame_stats['class_counts'][cid]}"
                for cid in self.allowed_classes if frame_stats['class_counts'][cid] > 0
            ])
            if class_summary:
                print(f"Frame {frame_number}: {frame_stats['kept']}/{frame_stats['total']} kept ({class_summary})")
            else:
                print(f"Frame {frame_number}: {frame_stats['kept']}/{frame_stats['total']} kept (no players or ball detected)")
        
        # Store in MOTRv2 format
        frame_path = self.get_frame_path(sequence_name, frame_number)
        self.detection_database[frame_path] = frame_detections
        
        return {frame_path: frame_detections}
    
    def save_database(self, output_path: str, filename: str = "det_db_motrv2.json") -> str:
        """Save the detection database in MOTRv2 format."""
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(self.detection_database, f, indent=2)
        
        # Print comprehensive statistics
        print("\n=== MOTRv2 Detection Database Summary ===")
        print(f"Total detections processed: {self.stats['total_detections']}")
        print(f"Filtered by score threshold: {self.stats['filtered_by_score']}")
        print(f"Filtered by class restrictions: {self.stats['filtered_by_translation']}")
        print(f"Filtered by court area: {self.stats['filtered_by_court']}")
        print(f"Final detections kept: {self.stats['kept_detections']}")
        print(f"Retention rate: {self.stats['kept_detections']/max(1, self.stats['total_detections']):.1%}")
        
        print("\n=== Beach Volleyball Class Breakdown ===")
        for class_id in self.allowed_classes:
            class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")
            count = self.stats['class_counts'][class_id]
            percentage = count / max(1, self.stats['kept_detections']) * 100
            print(f"{class_name}: {count} detections ({percentage:.1f}%)")
        
        print(f"\n=== MOTRv2 Format Validation ===")
        print(f"Total frames processed: {len(self.detection_database)}")
        
        # Show sample frame paths for verification
        sample_paths = list(self.detection_database.keys())[:3]
        for path in sample_paths:
            det_count = len(self.detection_database[path])
            print(f"Sample frame: {path} -> {det_count} detections")
            if det_count > 0:
                # Show first detection string format
                first_det = self.detection_database[path][0].strip()
                print(f"  Format example: {first_det}")
        
        print(f"\nMOTRv2 detection database saved to: {full_path}")
        print("✅ Ready for MOTRv2 tracking!")
        
        return full_path
    
    def reset(self):
        """Reset the detection database and statistics."""
        self.detection_database = {}
        self.stats = {
            'total_detections': 0,
            'filtered_by_score': 0,
            'filtered_by_court': 0,
            'filtered_by_translation': 0,
            'kept_detections': 0,
            'class_counts': {class_id: 0 for class_id in self.allowed_classes}
        }


class MOTRv2VideoProcessor:
    """Compatibility class for video processing."""
    
    def __init__(self, formatter: MOTRv2Formatter):
        self.formatter = formatter
    
    def process_video(self, video_path: str, sequence_name: str = None):
        """Process entire video with the formatter."""
        if sequence_name is None:
            sequence_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing video: {video_path}")
        print("Note: Use torch_inf.py for actual video processing with D-FINE model")
        return sequence_name