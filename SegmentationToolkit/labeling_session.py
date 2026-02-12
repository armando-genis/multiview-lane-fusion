"""Labeling session management."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from tkinter import messagebox

from mask_generator import MaskGenerator
from image_processor import ImageProcessor
from file_ops import FileManager
from gui import MaskViewer


class LabelingSession:
    """Manages a labeling session with proper state handling."""
    
    def __init__(
        self,
        image_dir: Path,
        config: Dict,
        config_path: Path,
        device: str = "cuda"
    ):
        """
        Initialize labeling session.
        
        Args:
            image_dir: Directory containing input images
            config: Configuration dictionary
            config_path: Path to config file
            device: Device to use ("cuda" or "cpu")
        """
        # Extract configuration
        classes = config['ontology']['classes']
        self.prompts = [cls['prompt'] for cls in classes]
        self.class_names = [cls['name'] for cls in classes]
        self.class_colors = {cls['name']: cls.get('color', [255, 0, 0]) for cls in classes}
        
        model_config = config.get('model', {})
        self.max_image_dim = model_config.get('max_image_dim', 1024)
        self.mask_opacity = model_config.get('mask_opacity', 0.15)
        self.border_width = model_config.get('border_width', 1)
        
        # Store device and config for processor initialization
        self.device = device
        self.config = config
        self.config_path = config_path
        
        # Initialize components
        self.file_manager = FileManager(image_dir.parent)
        self._initialize_processors()
        
        # Get image files
        self.image_files = self.file_manager.get_image_files(image_dir)
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        # State
        self.current_idx = 0
        self.current_image_data = None  # Cache for (image, class_masks, inference_time, info)
        
        # Initialize GUI
        self.viewer = MaskViewer("Segmentation Mask Viewer")
        self._load_ontology_to_editor()
        self._setup_gui_callbacks()
        
        print(f"Found {len(self.image_files)} images")
        print(f"Accepted images: {self.file_manager.accepted_images_dir}")
        print(f"Accepted masks: {self.file_manager.accepted_masks_dir}")
        print(f"Discarded images: {self.file_manager.discarded_dir}")
    
    def _initialize_processors(self):
        """Initialize mask generator and image processor."""
        self.mask_generator = MaskGenerator(
            prompts=self.prompts, 
            device=self.device, 
            max_image_dim=self.max_image_dim
        )
        self.image_processor = ImageProcessor(
            mask_generator=self.mask_generator,
            file_manager=self.file_manager,
            prompts=self.prompts,
            class_names=self.class_names,
            class_colors=self.class_colors,
            mask_opacity=self.mask_opacity,
            border_width=self.border_width
        )
    
    def _load_ontology_to_editor(self):
        """Load ontology YAML into the GUI editor."""
        if self.config_path:
            with open(self.config_path, 'r') as f:
                ontology_yaml = f.read()
            self.viewer.set_ontology_text(ontology_yaml)
    
    def _setup_gui_callbacks(self):
        """Setup GUI callbacks."""
        self.viewer.on_accept = self.accept_current_image
        self.viewer.on_discard = self.discard_current_image
        self.viewer.on_reprocess = self.reprocess_current_image
        self.viewer.on_quit = self.quit
        self.viewer.on_ontology_change = self.update_ontology
    
    def get_current_image_file(self) -> Optional[Path]:
        """Get current image file."""
        if self.current_idx >= len(self.image_files):
            return None
        return self.image_files[self.current_idx]
    
    def process_current_image(self):
        """Process and display current image."""
        image_file = self.get_current_image_file()
        if image_file is None:
            self.viewer.update_status("No more images to process")
            return
        
        print(f"\n[{self.current_idx + 1}/{len(self.image_files)}] Processing: {image_file.name}")
        self.viewer.update_status(f"Processing: {image_file.name} ({self.current_idx + 1}/{len(self.image_files)})")
        
        try:
            image, class_masks, inference_time, info = self.image_processor.process_image(image_file)
            
            # Cache the results for accept/discard operations
            self.current_image_data = (image, class_masks, inference_time, info)
            
            self.viewer.display_image(image, class_masks, info)
            self.viewer.update_status(f"Ready: {image_file.name} ({self.current_idx + 1}/{len(self.image_files)})")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()
            self.viewer.update_status(f"Error: {e}")
    
    def accept_current_image(self):
        """Accept current image: save mask and move to accepted folder."""
        image_file = self.get_current_image_file()
        if image_file is None or self.current_image_data is None:
            return
        
        try:
            # Use cached masks instead of re-running the model
            _, class_masks, _, _ = self.current_image_data
            
            image_dest_path, mask_path = self.image_processor.save_and_accept_image(
                image_file, class_masks
            )
            print(f"✓ Accepted: {image_file.name} -> {image_dest_path}")
            print(f"  Mask saved: {mask_path}")
            
            # Clear cache
            self.current_image_data = None
            
            # Remove from list and move to next
            self.image_files.pop(self.current_idx)
            self._advance_to_next()
        except Exception as e:
            print(f"Error accepting image: {e}")
            import traceback
            traceback.print_exc()
            self.viewer.update_status(f"Error: {e}")
    
    def discard_current_image(self):
        """Discard current image: move to discarded folder."""
        image_file = self.get_current_image_file()
        if image_file is None:
            return
        
        try:
            dest_path = self.image_processor.discard_image(image_file)
            print(f"✗ Discarded: {image_file.name} -> {dest_path}")
            
            # Remove from list and move to next
            self.image_files.pop(self.current_idx)
            self._advance_to_next()
        except Exception as e:
            print(f"Error discarding image: {e}")
            import traceback
            traceback.print_exc()
            self.viewer.update_status(f"Error: {e}")
    
    def reprocess_current_image(self):
        """Reprocess current image with current ontology."""
        self.process_current_image()
    
    def _advance_to_next(self):
        """Advance to next image or finish."""
        if self.image_files:
            self.process_current_image()
        else:
            self.viewer.update_status("All images processed!")
            print("\n✓ All images processed! Raw images folder is now empty.")
            messagebox.showinfo("Complete", "All images have been processed!\nRaw images folder is now empty.")
    
    def update_ontology(self, ontology_yaml: str):
        """Update ontology from YAML text."""
        try:
            new_config = yaml.safe_load(ontology_yaml)
            
            if 'ontology' not in new_config or 'classes' not in new_config['ontology']:
                raise ValueError("Invalid ontology structure: missing 'ontology.classes'")
            
            # Extract new configuration
            new_classes = new_config['ontology']['classes']
            new_prompts = [cls['prompt'] for cls in new_classes]
            new_class_names = [cls['name'] for cls in new_classes]
            new_class_colors = {cls['name']: cls.get('color', [255, 0, 0]) for cls in new_classes}
            
            # Update model config
            if 'model' in new_config:
                model_config = new_config.get('model', {})
                self.max_image_dim = model_config.get('max_image_dim', 1024)
                self.mask_opacity = model_config.get('mask_opacity', 0.15)
                self.border_width = model_config.get('border_width', 1)
            
            # Update state
            self.config = new_config
            self.prompts = new_prompts
            self.class_names = new_class_names
            self.class_colors = new_class_colors
            self.current_image_data = None  # Clear cache since ontology changed
            
            # Reinitialize processors with new configuration
            print("\nUpdating ontology...")
            print(f"New prompts: {new_prompts}")
            self._initialize_processors()
            
            print("✓ Ontology updated successfully")
            self.viewer.update_status("Ontology updated - reprocess current image to see changes")
            self.reprocess_current_image()
            
        except Exception as e:
            print(f"Error updating ontology: {e}")
            import traceback
            traceback.print_exc()
            self.viewer.update_status(f"Error updating ontology: {e}")
    
    def quit(self):
        """Quit the labeling session."""
        remaining = len(self.image_files)
        if remaining > 0:
            response = messagebox.askyesno(
                "Quit?",
                f"There are {remaining} images remaining.\nAre you sure you want to quit?"
            )
            if not response:
                return
        self.viewer.destroy()
        print(f"\nProcessed {self.current_idx} images, {remaining} remaining")
    
    def run(self):
        """Start the labeling session."""
        self.process_current_image()
        self.viewer.run()
