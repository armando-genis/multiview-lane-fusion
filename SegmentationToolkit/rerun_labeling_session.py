"""Rerun-based labeling session."""

import yaml
from pathlib import Path
from typing import Dict, Optional
import threading
import time

try:
    from .mask_generator import MaskGenerator
    from .image_processor import ImageProcessor
    from .file_ops import FileManager
    from .rerun_viewer import RerunViewer
    from .ontology_editor import OntologyEditor
    from .control_panel import ControlPanel
except ImportError:
    from mask_generator import MaskGenerator
    from image_processor import ImageProcessor
    from file_ops import FileManager
    from rerun_viewer import RerunViewer
    from ontology_editor import OntologyEditor
    from control_panel import ControlPanel


class RerunLabelingSession:
    """Labeling session using Rerun for visualization."""
    
    def __init__(self, image_dir: Path, config: Dict, config_path: Path, device: str = "cuda"):
        classes = config['ontology']['classes']
        self.prompts = [cls['prompt'] for cls in classes]
        self.class_names = [cls['name'] for cls in classes]
        self.class_colors = {cls['name']: cls.get('color', [255, 0, 0]) for cls in classes}
        
        model_config = config.get('model', {})
        self.max_image_dim = model_config.get('max_image_dim', 1024)
        self.mask_opacity = model_config.get('mask_opacity', 0.15)
        self.border_width = model_config.get('border_width', 1)
        
        self.device = device
        self.config = config
        self.config_path = config_path
        
        self.file_manager = FileManager(image_dir.parent)
        self._initialize_processors()
        
        self.image_files = self.file_manager.get_image_files(image_dir)
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        self.current_idx = 0
        self.current_image_data = None
        self.running = True
        
        self.viewer = RerunViewer("segmentation_labeling", config_path=self.config_path)
        self.control_panel = ControlPanel()
        self.editor = None
        
        # Setup control panel callbacks
        self.control_panel.on_accept = self._handle_accept
        self.control_panel.on_discard = self._handle_discard
        self.control_panel.on_reprocess = self._handle_reprocess
        self.control_panel.on_edit_ontology = self.open_editor
        self.control_panel.on_quit = self._handle_quit
        
        print(f"\nFound {len(self.image_files)} images")
        print(f"Accepted: {self.file_manager.accepted_images_dir}")
        print(f"Masks: {self.file_manager.accepted_masks_dir}")
        print(f"Discarded: {self.file_manager.discarded_dir}")
        print("\nControl panel opened. Use buttons or keyboard shortcuts.")
    
    def _initialize_processors(self):
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
    
    def get_current_image_file(self):
        if self.current_idx >= len(self.image_files):
            return None
        return self.image_files[self.current_idx]
    
    def process_current_image(self):
        image_file = self.get_current_image_file()
        if image_file is None:
            status_msg = "All images processed"
            self.viewer.update_status(status_msg)
            self.control_panel.update_status(status_msg, "green")
            self.control_panel.set_buttons_enabled(False)
            print("\nAll images processed!")
            return False
        
        processing_msg = f"Processing: {image_file.name} ({self.current_idx + 1}/{len(self.image_files)})"
        print(f"\n[{self.current_idx + 1}/{len(self.image_files)}] Processing: {image_file.name}")
        self.viewer.update_status(processing_msg)
        self.control_panel.update_status(processing_msg, "orange")
        self.control_panel.set_buttons_enabled(False)
        
        try:
            image, class_masks, inference_time, info = self.image_processor.process_image(image_file)
            self.current_image_data = (image, class_masks, inference_time, info)
            
            info['current_idx'] = self.current_idx
            info['total'] = len(self.image_files)
            info['filename'] = image_file.name
            
            self.viewer.display_image(image, class_masks, info)
            ready_msg = f"Ready: {image_file.name} ({self.current_idx + 1}/{len(self.image_files)})"
            self.viewer.update_status(ready_msg)
            self.control_panel.update_status(ready_msg, "green")
            self.control_panel.set_buttons_enabled(True)
            return True
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"Error processing {image_file.name}: {e}")
            self.viewer.update_status(error_msg)
            self.control_panel.update_status(error_msg, "red")
            self.control_panel.set_buttons_enabled(True)
            return False
    
    def accept_current_image(self):
        image_file = self.get_current_image_file()
        if image_file is None or self.current_image_data is None:
            return False
        
        try:
            _, class_masks, _, _ = self.current_image_data
            image_dest_path, mask_path = self.image_processor.save_and_accept_image(
                image_file, class_masks
            )
            print(f"Accepted: {image_file.name}")
            
            self.current_image_data = None
            self.current_idx += 1
            return True
        except Exception as e:
            print(f"Error accepting: {e}")
            return False
    
    def discard_current_image(self):
        image_file = self.get_current_image_file()
        if image_file is None:
            return False
        
        try:
            dest_path = self.image_processor.discard_image(image_file)
            print(f"Discarded: {image_file.name}")
            
            self.current_image_data = None
            self.current_idx += 1
            return True
        except Exception as e:
            print(f"Error discarding: {e}")
            self.control_panel.update_status(f"Error: {e}", "red")
            return False
    
    def reprocess_current_image(self):
        print("\nReprocessing with updated ontology...")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            classes = self.config['ontology']['classes']
            self.prompts = [cls['prompt'] for cls in classes]
            self.class_names = [cls['name'] for cls in classes]
            self.class_colors = {cls['name']: cls.get('color', [255, 0, 0]) for cls in classes}
            
            model_config = self.config.get('model', {})
            self.mask_opacity = model_config.get('mask_opacity', 0.15)
            self.border_width = model_config.get('border_width', 1)
            
            self._initialize_processors()
            
            print(f"Ontology updated: {self.class_names}")
            
            self.current_image_data = None
            self.process_current_image()
            
        except Exception as e:
            print(f"Error reprocessing: {e}")
            self.control_panel.update_status(f"Error: {e}", "red")
    
    def _handle_accept(self):
        """Handle accept button click."""
        if self.accept_current_image():
            self.process_current_image()
    
    def _handle_discard(self):
        """Handle discard button click."""
        if self.discard_current_image():
            self.process_current_image()
    
    def _handle_reprocess(self):
        """Handle reprocess button click."""
        self.reprocess_current_image()
    
    def _handle_quit(self):
        """Handle quit button click."""
        self.quit()
    
    def open_editor(self):
        """Open ontology editor in a separate thread."""
        def run_editor():
            self.editor = OntologyEditor(self.config_path)
            self.editor.run()
        
        editor_thread = threading.Thread(target=run_editor, daemon=True)
        editor_thread.start()
        print("\nOntology editor opened. After saving, click 'Reprocess'.")
    
    def quit(self):
        print(f"\nProcessed: {self.current_idx}/{len(self.image_files)}")
        print(f"Remaining: {len(self.image_files) - self.current_idx}")
        self.running = False
        self.control_panel.destroy()
    
    def run(self):
        """Run the labeling session with GUI controls."""
        if not self.process_current_image():
            return
        
        print("\nUse the control panel buttons or keyboard shortcuts.")
        
        # Run the control panel (this is blocking)
        try:
            self.control_panel.run()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.quit()

