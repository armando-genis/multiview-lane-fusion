"""Core mask generation functionality using Grounded SAM."""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, List
import torch

try:
    from autodistill_grounded_sam import GroundedSAM
    from autodistill.detection import DetectionOntology
except ImportError as e:
    raise ImportError(
        "Required packages not found! Install with: pip install autodistill autodistill-grounded-sam"
    ) from e


class MaskGenerator:
    """Generate segmentation masks using Grounded SAM."""
    
    def __init__(self, prompts: List[str], device: str = "cuda", max_image_dim: int = 1024):
        """
        Initialize the mask generator.
        
        Args:
            prompts: List of text prompts for segmentation
            device: Device to use ("cuda" for GPU, "cpu" for CPU)
            max_image_dim: Maximum dimension for image resizing (saves GPU memory)
        """
        self.prompts = prompts
        self.device = device
        self.max_image_dim = max_image_dim
        
        print(f"Initializing Grounded SAM on {device}...")
        print(f"Prompts: {prompts}")
        
        # Create ontology mapping prompts to class labels
        ontology = DetectionOntology([
            (prompt, prompt) for prompt in prompts
        ])
        
        # Initialize Grounded SAM model
        self.model = GroundedSAM(ontology=ontology)
    
    def generate_mask(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
        """
        Generate mask for a single image with per-class masks.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (original_image, class_masks_dict, inference_time)
            class_masks_dict: Dictionary mapping class names to binary masks
        """
        # Read and resize image
        image = self._read_and_resize_image(image_path)
        
        # Generate detections using Grounded SAM
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Run inference
            start_time = time.time()
            detections = self.model.predict(str(image_path))
            inference_time = time.time() - start_time
            print(f"  Inference time: {inference_time:.2f}s")
            
            # Process detections into per-class masks
            class_masks = self._process_detections(detections, image.shape[:2])
            
            # Print summary
            self._print_mask_summary(class_masks, image.shape[:2])
            
            return image, class_masks, inference_time
            
        except Exception as e:
            print(f"Error generating mask: {e}")
            import traceback
            traceback.print_exc()
            # Return empty masks on error
            empty_masks = self._create_empty_masks(image.shape[:2])
            return image, empty_masks, 0.0
    
    def _read_and_resize_image(self, image_path: Path) -> np.ndarray:
        """Read image and resize if needed to save GPU memory."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Resized image from {w}x{h} to {new_w}x{new_h} to save GPU memory")
        
        return image
    
    def _create_empty_masks(self, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Create empty masks for all prompts."""
        return {
            prompt: np.zeros(image_shape, dtype=np.uint8) 
            for prompt in self.prompts
        }
    
    def _process_detections(
        self, 
        detections, 
        image_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Process detections into per-class masks."""
        class_masks = self._create_empty_masks(image_shape)
        
        if not detections or detections.is_empty():
            return class_masks
        
        num_detections = len(detections.xyxy) if hasattr(detections, 'xyxy') and detections.xyxy is not None else 0
        print(f"  Processing {num_detections} detections")
        
        # Check what data we have
        has_class_ids = (hasattr(detections, 'class_id') and 
                        detections.class_id is not None and 
                        len(detections.class_id) > 0)
        
        has_masks = (hasattr(detections, 'mask') and 
                    detections.mask is not None and 
                    len(detections.mask) > 0)
        
        if not has_masks:
            return class_masks
        
        # Process each detection
        for i in range(num_detections):
            class_name = self._get_class_name_for_detection(i, detections, has_class_ids)
            det_mask = detections.mask[i]
            
            if det_mask is not None and isinstance(det_mask, np.ndarray):
                processed_mask = self._convert_detection_mask(det_mask, image_shape)
                class_masks[class_name] = np.maximum(class_masks[class_name], processed_mask)
                print(f"      ✓ Added mask for class '{class_name}' (detection {i})")
        
        return class_masks
    
    def _get_class_name_for_detection(
        self, 
        detection_idx: int, 
        detections, 
        has_class_ids: bool
    ) -> str:
        """Get class name for a detection."""
        if has_class_ids:
            class_id = int(detections.class_id[detection_idx])
            if class_id < len(self.prompts):
                return self.prompts[class_id]
            else:
                print(f"      Warning: class_id {class_id} out of range, using first class")
                return self.prompts[0]
        else:
            # No class info, assign to first class
            return self.prompts[0]
    
    def _convert_detection_mask(
        self, 
        det_mask: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Convert detection mask to uint8 format and resize if needed."""
        # Handle boolean masks
        if det_mask.dtype == bool:
            det_mask = det_mask.astype(np.uint8) * 255
        # Handle float masks (0-1 range)
        elif det_mask.dtype in [np.float32, np.float64]:
            if det_mask.max() <= 1.0:
                det_mask = (det_mask * 255).astype(np.uint8)
            else:
                det_mask = det_mask.astype(np.uint8)
        
        # Ensure mask is same size as image
        if det_mask.shape[:2] != target_shape:
            det_mask = cv2.resize(
                det_mask,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        return det_mask
    
    def _print_mask_summary(
        self, 
        class_masks: Dict[str, np.ndarray], 
        image_shape: Tuple[int, int]
    ):
        """Print summary of mask coverage."""
        total_pixels = 0
        total_image_pixels = image_shape[0] * image_shape[1]
        
        for class_name, mask in class_masks.items():
            pixels = np.sum(mask > 0)
            total_pixels += pixels
            if pixels > 0:
                print(f"  {class_name}: {pixels} pixels ({100 * pixels / mask.size:.1f}%)")
        
        print(f"  Total mask pixels: {total_pixels} ({100 * total_pixels / total_image_pixels:.1f}%)")
