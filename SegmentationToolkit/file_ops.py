"""File system operations for image and mask management."""

import cv2
import numpy as np
import random
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class FileManager:
    """Manages file operations for images and masks."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize file manager.
        
        Args:
            base_dir: Base directory (typically where raw_images is located)
        """
        self.base_dir = Path(base_dir)
        
        # Define directory structure
        # raw_images at data root, labeling and training are separate
        self.raw_images_dir = self.base_dir / "raw_images"
        self.accepted_images_dir = self.base_dir / "labeling" / "accepted" / "images"
        self.accepted_masks_dir = self.base_dir / "labeling" / "accepted" / "masks"
        self.discarded_dir = self.base_dir / "labeling" / "discarded"
        
        # Training directories (for future use by train.py)
        self.training_train_images_dir = self.base_dir / "training" / "train" / "images"
        self.training_train_masks_dir = self.base_dir / "training" / "train" / "masks"
        self.training_val_images_dir = self.base_dir / "training" / "val" / "images"
        self.training_val_masks_dir = self.base_dir / "training" / "val" / "masks"
        self.training_test_images_dir = self.base_dir / "training" / "test" / "images"
        self.training_test_masks_dir = self.base_dir / "training" / "test" / "masks"
        
        # Create directories
        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_images_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_masks_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self, image_dir: Optional[Path] = None) -> List[Path]:
        """
        Get all image files from a directory.
        
        Args:
            image_dir: Directory to scan (defaults to raw_images_dir)
            
        Returns:
            Sorted list of image file paths
        """
        if image_dir is None:
            image_dir = self.raw_images_dir
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            return []
        
        image_files = sorted(
            list(image_dir.glob("*.png")) + 
            list(image_dir.glob("*.jpg")) + 
            list(image_dir.glob("*.jpeg"))
        )
        return image_files
    
    def save_masks(
        self, 
        image_file: Path, 
        class_masks: Dict[str, np.ndarray],
        class_names: List[str],
        unique_id: Optional[str] = None
    ) -> Path:
        """
        Save combined mask for an image in SuperGradients-compatible format.
        
        Saves a single-channel PNG mask where pixel values are class IDs:
        - 0 = background
        - 1 = first class in ontology
        - 2 = second class in ontology
        - etc.
        
        Args:
            image_file: Path to the original image file
            class_masks: Dictionary mapping class names to binary masks
            class_names: List of class names in order (defines class ID mapping)
            unique_id: Optional UUID to append to filename (if None, uses image stem only)
            
        Returns:
            Path to the saved combined mask file
        """
        # Get image dimensions from first mask
        if not class_masks:
            raise ValueError("No masks provided")
        
        first_mask = next(iter(class_masks.values()))
        image_shape = first_mask.shape[:2]
        
        # Create combined mask with class IDs
        # Background = 0, classes = 1, 2, 3, ...
        combined_mask = np.zeros(image_shape, dtype=np.uint8)
        for idx, class_name in enumerate(class_names):
            if class_name in class_masks:
                # Class ID = index + 1 (0 is reserved for background)
                combined_mask[class_masks[class_name] > 0] = idx + 1
        
        # Save as single-channel PNG (SuperGradients format)
        if unique_id:
            mask_filename = f"{image_file.stem}_{unique_id}_mask.png"
        else:
            mask_filename = image_file.stem + "_mask.png"
        combined_mask_path = self.accepted_masks_dir / mask_filename
        
        # Use cv2.imwrite with single-channel format
        # This ensures it's saved as grayscale PNG with class IDs
        cv2.imwrite(str(combined_mask_path), combined_mask)
        
        return combined_mask_path
    
    def accept_image(self, image_file: Path, unique_id: Optional[str] = None) -> Path:
        """
        Move image to accepted/images directory with UUID to prevent overwrites.
        
        Args:
            image_file: Path to image file to move
            unique_id: Optional UUID to append to filename (if None, generates new one)
            
        Returns:
            Path to the new location
        """
        # Generate unique filename with UUID to prevent overwrites
        if unique_id is None:
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        stem = image_file.stem
        suffix = image_file.suffix
        unique_filename = f"{stem}_{unique_id}{suffix}"
        dest_path = self.accepted_images_dir / unique_filename
        image_file.rename(dest_path)
        return dest_path
    
    def discard_image(self, image_file: Path) -> Path:
        """
        Move image to discarded directory.
        
        Args:
            image_file: Path to image file to move
            
        Returns:
            Path to the new location
        """
        dest_path = self.discarded_dir / image_file.name
        image_file.rename(dest_path)
        return dest_path
    
    def read_image(self, image_path: Path) -> np.ndarray:
        """
        Read an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image
    
    def prepare_training_data(
        self,
        training_dir: Path,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[Path, Path, Path, Path]:
        """
        Prepare training data by splitting accepted data into train/val/test.
        
        Args:
            training_dir: Base directory for training data
            train_split: Training split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            Tuple of (train_images_dir, train_masks_dir, val_images_dir, val_masks_dir)
        """
        print("\nPreparing training data splits...")
        
        # Define output directories
        train_images_dir = training_dir / "train" / "images"
        train_masks_dir = training_dir / "train" / "masks"
        val_images_dir = training_dir / "val" / "images"
        val_masks_dir = training_dir / "val" / "masks"
        test_images_dir = training_dir / "test" / "images"
        test_masks_dir = training_dir / "test" / "masks"
        
        # Clear existing training data
        print("Clearing existing training data...")
        for dir_path in [train_images_dir, train_masks_dir, val_images_dir, 
                         val_masks_dir, test_images_dir, test_masks_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image-mask pairs
        image_files = sorted(list(self.accepted_images_dir.glob("*.png")) + 
                            list(self.accepted_images_dir.glob("*.jpg")))
        
        # Match images with masks
        pairs = []
        for img_path in image_files:
            # Find corresponding mask (should have _mask.png suffix before extension)
            mask_name = img_path.stem + "_mask.png"
            mask_path = self.accepted_masks_dir / mask_name
            
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        
        if not pairs:
            raise ValueError("No image-mask pairs found!")
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        # Split data
        n = len(pairs)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        
        # Copy files
        def copy_pairs(pair_list, img_dir, mask_dir):
            for img_path, mask_path in pair_list:
                shutil.copy(img_path, img_dir / img_path.name)
                shutil.copy(mask_path, mask_dir / mask_path.name)
        
        print(f"Copying {len(train_pairs)} pairs to train/")
        copy_pairs(train_pairs, train_images_dir, train_masks_dir)
        
        print(f"Copying {len(val_pairs)} pairs to val/")
        copy_pairs(val_pairs, val_images_dir, val_masks_dir)
        
        print(f"Copying {len(test_pairs)} pairs to test/")
        copy_pairs(test_pairs, test_images_dir, test_masks_dir)
        
        print(f"Data split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        return train_images_dir, train_masks_dir, val_images_dir, val_masks_dir




