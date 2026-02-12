"""Image processing for labeling."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

from mask_generator import MaskGenerator
from file_ops import FileManager


class ImageProcessor:
    """Processes images and handles mask generation/conversion."""
    
    def __init__(
        self,
        mask_generator: MaskGenerator,
        file_manager: FileManager,
        prompts: List[str],
        class_names: List[str],
        class_colors: Dict[str, List[int]],
        mask_opacity: float = 0.15,
        border_width: int = 1
    ):
        """
        Initialize image processor.
        
        Args:
            mask_generator: MaskGenerator instance
            file_manager: FileManager instance
            prompts: List of prompts for detection
            class_names: List of class names for display
            class_colors: Dictionary mapping class names to BGR colors
            mask_opacity: Mask overlay opacity
            border_width: Mask border width
        """
        self.mask_generator = mask_generator
        self.file_manager = file_manager
        self.prompts = prompts
        self.class_names = class_names
        self.class_colors = class_colors
        self.mask_opacity = mask_opacity
        self.border_width = border_width
        
        # Create prompt to class name mapping
        self.prompt_to_class = {
            prompt: class_name 
            for prompt, class_name in zip(prompts, class_names)
        }
    
    def _convert_prompt_masks_to_class_masks(
        self, 
        prompt_masks: Dict[str, np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Convert prompt-based masks to class-name-based masks.
        
        Args:
            prompt_masks: Masks keyed by prompts
            image_shape: (height, width) of image
            
        Returns:
            Masks keyed by class names
        """
        class_masks = {}
        for prompt, class_name in self.prompt_to_class.items():
            if prompt in prompt_masks:
                class_masks[class_name] = prompt_masks[prompt]
            else:
                class_masks[class_name] = np.zeros(image_shape, dtype=np.uint8)
        return class_masks
    
    def process_image(
        self, 
        image_file: Path
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Dict]:
        """
        Process an image and generate masks.
        
        Args:
            image_file: Path to image file
            
        Returns:
            Tuple of (image, class_masks, inference_time, info_dict)
        """
        # Generate masks
        image, prompt_masks, inference_time = self.mask_generator.generate_mask(image_file)
        
        # Convert to class-based masks
        class_masks = self._convert_prompt_masks_to_class_masks(
            prompt_masks, 
            (image.shape[0], image.shape[1])
        )
        
        # Prepare info dictionary
        info = {
            'image_name': image_file.name,
            'prompts': self.prompts,
            'class_names': self.class_names,
            'inference_time': inference_time,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'image_shape': (image.shape[0], image.shape[1], image.shape[2]),
            'class_colors': self.class_colors,
            'mask_opacity': self.mask_opacity,
            'border_width': self.border_width,
        }
        
        return image, class_masks, inference_time, info
    
    def save_and_accept_image(
        self, 
        image_file: Path, 
        class_masks: Dict[str, np.ndarray]
    ) -> Tuple[Path, Path]:
        """
        Save masks and move image to accepted folder with UUID to prevent overwrites.
        
        Args:
            image_file: Path to image file
            class_masks: Pre-computed class masks
            
        Returns:
            Tuple of (image_dest_path, combined_mask_path)
        """
        # Generate UUID for this image/mask pair
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        
        # Save combined mask with same UUID
        combined_mask_path = self.file_manager.save_masks(
            image_file, class_masks, self.class_names, unique_id=unique_id
        )
        
        # Move image to accepted folder with same UUID
        image_dest_path = self.file_manager.accept_image(image_file, unique_id=unique_id)
        
        return image_dest_path, combined_mask_path
    
    def discard_image(self, image_file: Path) -> Path:
        """
        Discard image: move to discarded folder.
        
        Args:
            image_file: Path to image file
            
        Returns:
            Path to discarded image
        """
        return self.file_manager.discard_image(image_file)
