"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
MOTRv2 Integration Module for D-FINE - Fixed version with proper tensor handling
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np


class MOTRv2Formatter:
    """
    Formatter to convert D-FINE detection outputs to MOTRv2 compatible JSON format.
    
    This formatter properly handles PyTorch tensors by detaching them from the
    computation graph before conversion to NumPy arrays. This is essential for
    inference mode where we don't need gradient tracking.
    """
    
    def __init__(self, 
                 score_threshold: float = 0.3,
                 category_mapping: Optional[Dict[int, int]] = None):
        """
        Initialize the MOTRv2 formatter.
        
        Args:
            score_threshold: Minimum confidence score to include a detection
            category_mapping: Optional mapping from D-FINE class IDs to MOTRv2 category IDs
                            If None, uses identity mapping
        """
        self.score_threshold = score_threshold
        self.category_mapping = category_mapping or {}
        self.detection_database = {}
        
    def convert_bbox_format(self, bbox: torch.Tensor) -> List[float]:
        """
        Convert from D-FINE's corner format [x1, y1, x2, y2] to MOTRv2's [x, y, width, height].
        
        This function properly handles both torch tensors and numpy arrays,
        ensuring gradient safety by detaching tensors before conversion.
        
        Args:
            bbox: Tensor of shape (4,) with [x1, y1, x2, y2]
            
        Returns:
            List with [x, y, width, height] format
        """
        # Handle both tensor and numpy array inputs safely
        if isinstance(bbox, torch.Tensor):
            # Detach from computation graph first, then move to CPU and convert
            bbox = bbox.detach().cpu().numpy()
            
        x1, y1, x2, y2 = bbox
        
        # Calculate width and height
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Use minimum x,y as top-left corner (handles coordinate swaps)
        x = min(x1, x2)
        y = min(y1, y2)
        
        return [float(x), float(y), float(width), float(height)]
    
    def process_single_frame(self, 
                           labels: torch.Tensor,
                           boxes: torch.Tensor, 
                           scores: torch.Tensor,
                           sequence_name: str,
                           frame_number: int) -> Dict[str, Any]:
        """
        Process D-FINE outputs for a single frame and format for MOTRv2.
        
        This method safely handles tensors that may still be attached to the
        computation graph by properly detaching them before conversion.
        
        Args:
            labels: Tensor of shape (N,) with class indices
            boxes: Tensor of shape (N, 4) with bounding boxes in xyxy format
            scores: Tensor of shape (N,) with confidence scores
            sequence_name: Name of the video sequence
            frame_number: Frame index (1-based for MOTRv2 compatibility)
            
        Returns:
            Dictionary with frame detections in MOTRv2 format
        """
        # Safely convert tensors to numpy arrays
        # The .detach() call is crucial - it creates a new tensor that doesn't
        # require gradients and isn't part of the computation graph
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
            
        # Filter by confidence threshold
        valid_mask = scores >= self.score_threshold
        
        # Process valid detections
        frame_detections = []
        for i in range(len(labels)):
            if valid_mask[i]:
                # Apply category mapping if provided
                category_id = self.category_mapping.get(int(labels[i]), int(labels[i]))
                
                detection = {
                    "bbox": self.convert_bbox_format(boxes[i]),
                    "score": float(scores[i]),
                    "category_id": category_id
                }
                frame_detections.append(detection)
        
        # Store in the nested structure expected by MOTRv2
        if sequence_name not in self.detection_database:
            self.detection_database[sequence_name] = {}
            
        # MOTRv2 expects string frame numbers
        self.detection_database[sequence_name][str(frame_number)] = frame_detections
        
        return {sequence_name: {str(frame_number): frame_detections}}
    
    def save_database(self, output_path: str, filename: str = "det_db_motrv2.json") -> str:
        """
        Save the detection database to JSON file in MOTRv2 format.
        
        Args:
            output_path: Directory to save the file
            filename: Output filename (default: det_db_motrv2.json)
            
        Returns:
            Full path to the saved file
        """
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(self.detection_database, f, indent=2)
            
        print(f"Saved MOTRv2 detection database to: {full_path}")
        print(f"Total sequences: {len(self.detection_database)}")
        total_frames = sum(len(frames) for frames in self.detection_database.values())
        print(f"Total frames: {total_frames}")
        
        return full_path
    
    def reset(self):
        """Reset the detection database for processing new sequences."""
        self.detection_database = {}
        
    def get_current_detections(self) -> Dict[str, Any]:
        """Get the current detection database."""
        return self.detection_database


class MOTRv2VideoProcessor:
    """
    Helper class to process entire video sequences with D-FINE and format for MOTRv2.
    This handles the video-level logic while MOTRv2Formatter handles frame-level formatting.
    """
    
    def __init__(self, 
                 formatter: MOTRv2Formatter,
                 default_sequence_name: str = "default_sequence"):
        """
        Initialize the video processor.
        
        Args:
            formatter: MOTRv2Formatter instance
            default_sequence_name: Default name for sequences when not specified
        """
        self.formatter = formatter
        self.default_sequence_name = default_sequence_name
        
    def process_detections_batch(self,
                               all_labels: List[torch.Tensor],
                               all_boxes: List[torch.Tensor],
                               all_scores: List[torch.Tensor],
                               sequence_name: Optional[str] = None,
                               start_frame: int = 1) -> Dict[str, Any]:
        """
        Process a batch of detections from multiple frames.
        
        This method ensures all tensors are properly detached from the
        computation graph before processing.
        
        Args:
            all_labels: List of label tensors, one per frame
            all_boxes: List of box tensors, one per frame
            all_scores: List of score tensors, one per frame
            sequence_name: Optional sequence name
            start_frame: Starting frame number (1-based)
            
        Returns:
            Complete detection database for the sequence
        """
        if sequence_name is None:
            sequence_name = self.default_sequence_name
            
        for frame_idx, (labels, boxes, scores) in enumerate(zip(all_labels, all_boxes, all_scores)):
            frame_number = start_frame + frame_idx
            self.formatter.process_single_frame(
                labels, boxes, scores, sequence_name, frame_number
            )
            
        return self.formatter.get_current_detections()