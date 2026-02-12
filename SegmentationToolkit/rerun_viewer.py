"""Simple Rerun viewer for segmentation labeling."""

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, Optional, List


class RerunViewer:
    """Minimalist Rerun viewer for segmentation visualization."""
    
    def __init__(self, app_id: str = "segmentation_labeling", config_path: Optional[Path] = None):
        rr.init(app_id, spawn=True)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        
        # Load class names from config
        self.class_names = []
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.class_names = [cls['name'] for cls in config['ontology']['classes']]
        
        self._setup_blueprint()
    
    def _setup_blueprint(self):
        """Setup custom 3-column layout."""
        # Create mask views for each class
        mask_views = []
        if self.class_names:
            for class_name in self.class_names:
                mask_views.append(
                    rrb.Spatial2DView(origin=f"masks/{class_name}", name=f"Mask: {class_name.capitalize()}")
                )
            row_shares = [1] * len(mask_views)
        else:
            # Fallback if no classes loaded
            mask_views.append(rrb.Spatial2DView(origin="masks", name="Individual Masks"))
            row_shares = [1]
        
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                # Column 1: Original and Segmented images (2 rows)
                rrb.Vertical(
                    rrb.Spatial2DView(origin="image/original", name="Original Image"),
                    rrb.Spatial2DView(origin="image/overlay", name="Segmented Image"),
                    row_shares=[1, 1]
                ),
                # Column 2: Individual masks (one row per class)
                rrb.Vertical(
                    *mask_views,
                    row_shares=row_shares
                ),
                # Column 3: Info and status
                rrb.Vertical(
                    rrb.TextDocumentView(origin="info", name="Information"),
                    rrb.TextDocumentView(origin="status", name="Status"),
                    row_shares=[3, 1]
                ),
                column_shares=[2, 2, 1]
            ),
            collapse_panels=True
        )
        rr.send_blueprint(blueprint)
    
    def display_image(self, image: np.ndarray, class_masks: Dict[str, np.ndarray], info: Dict):
        """Display image with masks in Rerun."""
        if 'current_idx' in info:
            rr.set_time_sequence("image_index", info['current_idx'])
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Log original image
        rr.log("image/original", rr.Image(image_rgb))
        
        # Create and log overlay
        overlay = self._create_overlay(image_rgb, class_masks, info)
        rr.log("image/overlay", rr.Image(overlay))
        
        # Log individual masks
        classes_detected = []
        for class_name, mask in class_masks.items():
            if mask is not None and np.any(mask):
                classes_detected.append(class_name)
                mask_viz = self._create_mask_viz(image_rgb, mask, info.get('class_colors', {}).get(class_name, [255, 0, 0]))
                rr.log(f"masks/{class_name}", rr.Image(mask_viz))
                
                # Log statistics
                coverage = (np.sum(mask) / mask.size) * 100
                rr.log(f"stats/{class_name}", rr.Scalar(coverage))
        
        info['classes_detected'] = classes_detected
        
        # Log info
        rr.log("info", rr.TextDocument(self._format_info(info), media_type=rr.MediaType.TEXT))
    
    def _create_overlay(self, image_rgb: np.ndarray, class_masks: Dict[str, np.ndarray], info: Dict) -> np.ndarray:
        """Create overlay with all masks."""
        overlay = image_rgb.copy().astype(np.float32)
        opacity = info.get('mask_opacity', 0.15)
        border = info.get('border_width', 1)
        
        for class_name, mask in class_masks.items():
            if mask is None or not np.any(mask):
                continue
            
            color = info.get('class_colors', {}).get(class_name, [255, 0, 0])
            color_rgb = [color[2], color[1], color[0]]  # BGR to RGB
            
            mask_3d = np.stack([mask] * 3, axis=-1)
            overlay = np.where(mask_3d, 
                             overlay * (1 - opacity) + np.array(color_rgb) * opacity,
                             overlay)
            
            if border > 0:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color_rgb, border)
        
        return overlay.astype(np.uint8)
    
    def _create_mask_viz(self, image_rgb: np.ndarray, mask: np.ndarray, color: list) -> np.ndarray:
        """Create visualization for single mask."""
        viz = image_rgb.copy().astype(np.float32)
        color_rgb = [color[2], color[1], color[0]]
        
        mask_3d = np.stack([mask] * 3, axis=-1)
        viz = np.where(mask_3d, 
                      viz * 0.5 + np.array(color_rgb) * 0.5,
                      viz * 0.3)
        
        return viz.astype(np.uint8)
    
    def _format_info(self, info: Dict) -> str:
        """Format information as text."""
        lines = ["=" * 50]
        lines.append("IMAGE INFORMATION")
        lines.append("=" * 50)
        
        if 'filename' in info:
            lines.append(f"File: {info['filename']}")
        if 'current_idx' in info:
            lines.append(f"Progress: {info['current_idx'] + 1}/{info.get('total', '?')}")
        if 'image_shape' in info:
            lines.append(f"Shape: {info['image_shape']}")
        if 'inference_time' in info:
            lines.append(f"Inference: {info['inference_time']:.2f}s")
        if 'classes_detected' in info:
            lines.append(f"Classes: {', '.join(info['classes_detected'])}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def update_status(self, status: str):
        """Update status message."""
        rr.log("status", rr.TextDocument(status, media_type=rr.MediaType.TEXT))
