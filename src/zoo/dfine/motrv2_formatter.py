"""
Enhanced MOTRv2 Formatter with Object ID Translation and Court Area Filtering
This module provides comprehensive filtering and translation for beach volleyball analysis
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
    """
    Interactive tool for manually selecting the court area in a video frame.
    This allows users to define the region of interest where valid detections should occur.
    """
    
    def __init__(self):
        self.points = []
        self.court_bounds = None
        
    def select_court_area(self, image_path: str) -> Dict[str, float]:
        """
        Interactive method to select court boundaries by clicking on the image.
        
        Instructions:
        - Click to define the court corners (4 points)
        - Press 'r' to reset
        - Press 'q' to quit and save
        
        Returns:
            Dictionary with normalized court boundaries
        """
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
                        # Draw lines between points
                        ax.plot([self.points[-2][0], self.points[-1][0]], 
                               [self.points[-2][1], self.points[-1][1]], 'r-', linewidth=2)
                    
                    if len(self.points) == 4:
                        # Complete the rectangle
                        ax.plot([self.points[-1][0], self.points[0][0]], 
                               [self.points[-1][1], self.points[0][1]], 'r-', linewidth=2)
                        ax.set_title("Court area selected! Press 'q' to confirm or 'r' to reset")
                    
                    plt.draw()
        
        def onkey(event):
            if event.key == 'r':
                # Reset selection
                self.points = []
                ax.clear()
                ax.imshow(img_rgb)
                ax.set_title("Click 4 corners of the court area (or press 'c' to use center region)")
                plt.draw()
            elif event.key == 'c':
                # Use center 80% as default
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
            # Calculate bounding box from points
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


class EnhancedMOTRv2Formatter:
    """
    Enhanced formatter with Object ID translation and court area filtering.
    
    This formatter addresses two key challenges:
    1. Translating between Objects365 and COCO class IDs
    2. Filtering detections to only include those within the court area
    """
    
    # Object ID mapping from Objects365 to COCO
    OBJECTS365_TO_COCO = {
        0: 0,      # Person -> person
        240: 37,   # Volleyball/Ball -> sports ball
        # Add more mappings as needed for your use case
    }
    
    def __init__(self, 
                 score_threshold: float = 0.3,
                 court_bounds: Optional[Dict[str, float]] = None,
                 enable_translation: bool = True):
        """
        Initialize the enhanced formatter.
        
        Args:
            score_threshold: Minimum confidence score to include a detection
            court_bounds: Dictionary with 'left', 'right', 'top', 'bottom' as ratios (0-1)
            enable_translation: Whether to translate Objects365 IDs to COCO IDs
        """
        self.score_threshold = score_threshold
        self.court_bounds = court_bounds
        self.enable_translation = enable_translation
        self.detection_database = {}
        
        # Statistics for debugging
        self.stats = {
            'total_detections': 0,
            'filtered_by_score': 0,
            'filtered_by_court': 0,
            'filtered_by_translation': 0,
            'kept_detections': 0
        }
    
    def translate_class_id(self, objects365_id: int) -> Optional[int]:
        """
        Translate Objects365 class ID to COCO class ID.
        
        Returns None if no mapping exists (which filters out the detection).
        """
        if not self.enable_translation:
            return objects365_id
            
        coco_id = self.OBJECTS365_TO_COCO.get(objects365_id)
        
        if coco_id is None:
            # Log unmapped classes for debugging
            if objects365_id not in self.OBJECTS365_TO_COCO:
                print(f"Warning: No COCO mapping for Objects365 class {objects365_id}")
        
        return coco_id
    
    def is_in_court_area(self, bbox: List[float], image_width: int, image_height: int) -> bool:
        """
        Check if a detection is within the defined court boundaries.
        
        Args:
            bbox: Bounding box in [x, y, width, height] format
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            True if the detection center is within court bounds
        """
        if self.court_bounds is None:
            return True  # No court bounds defined, keep all detections
        
        # Calculate center of the bounding box
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        # Convert court bounds from ratios to pixels
        court_left = self.court_bounds['left'] * image_width
        court_right = self.court_bounds['right'] * image_width
        court_top = self.court_bounds['top'] * image_height
        court_bottom = self.court_bounds['bottom'] * image_height
        
        # Check if center is within bounds
        return (court_left <= center_x <= court_right and 
                court_top <= center_y <= court_bottom)
    
    def convert_bbox_format(self, bbox: torch.Tensor) -> List[float]:
        """Convert from corner format [x1, y1, x2, y2] to [x, y, width, height]."""
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
            
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        x = min(x1, x2)
        y = min(y1, y2)
        
        return [float(x), float(y), float(width), float(height)]
    
    def process_single_frame(self, 
                           labels: torch.Tensor,
                           boxes: torch.Tensor, 
                           scores: torch.Tensor,
                           sequence_name: str,
                           frame_number: int,
                           image_width: int,
                           image_height: int) -> Dict[str, Any]:
        """
        Process detections with translation and court filtering.
        
        This method applies a multi-stage filtering pipeline:
        1. Score threshold filtering
        2. Object ID translation (Objects365 -> COCO)
        3. Court area filtering
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
            'kept': 0
        }
        
        for i in range(len(labels)):
            self.stats['total_detections'] += 1
            
            # Stage 1: Score threshold
            if scores[i] < self.score_threshold:
                self.stats['filtered_by_score'] += 1
                frame_stats['score_filtered'] += 1
                continue
            
            # Stage 2: Class ID translation
            objects365_id = int(labels[i])
            coco_id = self.translate_class_id(objects365_id)
            
            if coco_id is None:
                self.stats['filtered_by_translation'] += 1
                frame_stats['translation_filtered'] += 1
                continue
            
            # Convert bbox format
            bbox = self.convert_bbox_format(boxes[i])
            
            # Stage 3: Court area filtering
            if not self.is_in_court_area(bbox, image_width, image_height):
                self.stats['filtered_by_court'] += 1
                frame_stats['court_filtered'] += 1
                continue
            
            # Detection passed all filters
            detection = {
                "bbox": bbox,
                "score": float(scores[i]),
                "category_id": coco_id  # Use translated ID
            }
            frame_detections.append(detection)
            self.stats['kept_detections'] += 1
            frame_stats['kept'] += 1
        
        # Log frame statistics
        if frame_number % 10 == 0:  # Log every 10th frame to avoid spam
            print(f"Frame {frame_number}: {frame_stats['kept']}/{frame_stats['total']} kept "
                  f"(score: -{frame_stats['score_filtered']}, "
                  f"translation: -{frame_stats['translation_filtered']}, "
                  f"court: -{frame_stats['court_filtered']})")
        
        # Store in database
        if sequence_name not in self.detection_database:
            self.detection_database[sequence_name] = {}
            
        self.detection_database[sequence_name][str(frame_number)] = frame_detections
        
        return {sequence_name: {str(frame_number): frame_detections}}
    
    def save_database(self, output_path: str, filename: str = "det_db_motrv2.json") -> str:
        """Save the detection database with filtering statistics."""
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(self.detection_database, f, indent=2)
        
        # Print comprehensive statistics
        print("\n=== Detection Processing Summary ===")
        print(f"Total detections processed: {self.stats['total_detections']}")
        print(f"Filtered by score threshold: {self.stats['filtered_by_score']}")
        print(f"Filtered by translation: {self.stats['filtered_by_translation']}")
        print(f"Filtered by court area: {self.stats['filtered_by_court']}")
        print(f"Final detections kept: {self.stats['kept_detections']}")
        print(f"Retention rate: {self.stats['kept_detections']/max(1, self.stats['total_detections']):.1%}")
        print(f"\nMOTRv2 detection database saved to: {full_path}")
        
        return full_path
    
    def reset(self):
        """Reset the detection database and statistics."""
        self.detection_database = {}
        self.stats = {
            'total_detections': 0,
            'filtered_by_score': 0,
            'filtered_by_court': 0,
            'filtered_by_translation': 0,
            'kept_detections': 0
        }