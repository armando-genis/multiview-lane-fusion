"""Tkinter-based GUI for viewing segmentation masks."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from typing import Dict, Optional, Callable, List
from pathlib import Path


class MaskViewer:
    """GUI viewer for displaying images with segmentation masks."""
    
    def __init__(self, window_title: str = "Segmentation Mask Viewer"):
        """
        Initialize the mask viewer GUI.
        
        Args:
            window_title: Title for the main window
        """
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1000x700")
        
        # Canvas default size
        self.canvas_width = 640
        self.canvas_height = 640
        
        # State
        self.current_image: Optional[np.ndarray] = None
        self.current_class_masks: Optional[Dict[str, np.ndarray]] = None
        self.current_info: Dict = {}
        self.on_accept: Optional[Callable] = None
        self.on_discard: Optional[Callable] = None
        self.on_reprocess: Optional[Callable] = None
        self.on_quit: Optional[Callable] = None
        self.on_ontology_change: Optional[Callable] = None
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Create and layout the UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Canvas for image with fixed default size
        self.canvas = tk.Canvas(image_frame, bg='gray', highlightthickness=0, 
                                width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Right panel (info + ontology editor)
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        
        # Info panel frame
        info_frame = ttk.LabelFrame(right_panel, text="Information", padding="5")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        info_frame.columnconfigure(0, weight=1)
        
        # Info text widget (smaller)
        self.info_text = tk.Text(info_frame, width=30, height=15, wrap=tk.WORD, state=tk.DISABLED, font=('TkDefaultFont', 9))
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.rowconfigure(0, weight=1)
        
        # Scrollbar for info text
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        # Ontology editor frame
        ontology_frame = ttk.LabelFrame(right_panel, text="Ontology Editor", padding="5")
        ontology_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        ontology_frame.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Ontology text editor
        self.ontology_text = tk.Text(ontology_frame, width=30, height=10, wrap=tk.WORD, font=('Courier', 9))
        self.ontology_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        ontology_frame.rowconfigure(0, weight=1)
        
        # Scrollbar for ontology editor
        ontology_scrollbar = ttk.Scrollbar(ontology_frame, orient="vertical", command=self.ontology_text.yview)
        ontology_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.ontology_text.configure(yscrollcommand=ontology_scrollbar.set)
        
        # Apply ontology button
        ttk.Button(ontology_frame, text="Apply Ontology", command=self._apply_ontology).grid(row=1, column=0, pady=(5, 0), sticky=(tk.W, tk.E))
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Accept", command=self._accept_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Discard", command=self._discard_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reprocess", command=self._reprocess_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Quit", command=self._quit).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _accept_image(self):
        """Trigger accept callback."""
        if self.on_accept:
            self.on_accept()
    
    def _discard_image(self):
        """Trigger discard callback."""
        if self.on_discard:
            self.on_discard()
    
    def _reprocess_image(self):
        """Trigger reprocess callback."""
        if self.on_reprocess:
            self.on_reprocess()
    
    def _quit(self):
        """Trigger quit callback."""
        if self.on_quit:
            self.on_quit()
        else:
            self.root.quit()
    
    def _apply_ontology(self):
        """Apply ontology changes from text editor."""
        if self.on_ontology_change:
            try:
                ontology_yaml = self.ontology_text.get(1.0, tk.END).strip()
                self.on_ontology_change(ontology_yaml)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply ontology: {e}")
    
    def set_ontology_text(self, ontology_yaml: str):
        """Set the ontology text in the editor."""
        self.ontology_text.delete(1.0, tk.END)
        self.ontology_text.insert(1.0, ontology_yaml)
    
    def display_image(self, image: np.ndarray, class_masks: Dict[str, np.ndarray], info: Dict):
        """
        Display an image with masks and information.
        
        Args:
            image: Original image (BGR format)
            class_masks: Dictionary mapping class names to binary masks
            info: Dictionary with information to display
        """
        self.current_image = image
        self.current_class_masks = class_masks
        self.current_info = info
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create overlay with masks
        overlay = self._create_overlay(image_rgb, class_masks)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(overlay)
        
        # Get canvas size (use configured size or actual size)
        canvas_width = max(self.canvas.winfo_width(), self.canvas_width)
        canvas_height = max(self.canvas.winfo_height(), self.canvas_height)
        
        # Calculate scaling to fit while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        if scale < 1.0:
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        
        # Center the image if it's smaller than canvas
        x_offset = (canvas_width - new_width) // 2 if new_width < canvas_width else 0
        y_offset = (canvas_height - new_height) // 2 if new_height < canvas_height else 0
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # Update info panel
        self._update_info_panel(info, class_masks)
    
    def _get_color_for_class(
        self,
        class_name: str,
        idx: int,
        class_colors_bgr: Dict[str, List[int]],
        default_colors_rgb: List[List[int]]
    ) -> List[int]:
        """
        Get RGB color for a class.
        
        Args:
            class_name: Name of the class
            idx: Index for default color fallback
            class_colors_bgr: Dictionary mapping class names to BGR colors
            default_colors_rgb: List of default RGB colors
            
        Returns:
            RGB color as list [R, G, B]
        """
        if class_name in class_colors_bgr:
            color_bgr = class_colors_bgr[class_name]
            if isinstance(color_bgr, list) and len(color_bgr) == 3:
                # Convert BGR to RGB
                return [color_bgr[2], color_bgr[1], color_bgr[0]]
        
        # Use default color from palette
        return default_colors_rgb[idx % len(default_colors_rgb)]
    
    def _create_overlay(self, image: np.ndarray, class_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create overlay image with colored masks.
        
        Args:
            image: RGB image
            class_masks: Dictionary mapping class names to binary masks
            
        Returns:
            Overlay image in RGB format
        """
        # Get colors from info (should be provided from config)
        class_colors_bgr = self.current_info.get('class_colors', {})
        
        # Default color palette if no colors provided (fallback)
        default_colors_rgb = [
            [255, 0, 0],      # Red
            [0, 0, 255],      # Blue
            [0, 255, 0],      # Green
            [255, 255, 0],    # Cyan
            [255, 0, 255],   # Magenta
            [0, 255, 255],    # Yellow
        ]
        
        # Create colored mask overlay (RGB format)
        colored_mask = np.zeros_like(image)
        
        for idx, (class_name, mask) in enumerate(class_masks.items()):
            if np.sum(mask > 0) > 0:
                color_rgb = self._get_color_for_class(class_name, idx, class_colors_bgr, default_colors_rgb)
                colored_mask[mask > 0] = color_rgb
        
        # Blend with original image
        opacity = self.current_info.get('mask_opacity', 0.15)
        overlay = cv2.addWeighted(image, 1.0 - opacity, colored_mask, opacity, 0)
        
        # Draw contours
        border_width = self.current_info.get('border_width', 1)
        for idx, (class_name, mask) in enumerate(class_masks.items()):
            if np.sum(mask > 0) > 0:
                color_rgb = self._get_color_for_class(class_name, idx, class_colors_bgr, default_colors_rgb)
                color_tuple = tuple(int(c) for c in color_rgb)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color_tuple, border_width)
        
        return overlay
    
    def _update_info_panel(self, info: Dict, class_masks: Dict[str, np.ndarray]):
        """Update the information panel with current image data."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        # Build info text
        lines = []
        lines.append(f"Image: {info.get('image_name', 'Unknown')}\n")
        lines.append(f"Prompts: {', '.join(info.get('prompts', []))}\n")
        lines.append(f"Inference time: {info.get('inference_time', 0):.2f}s\n")
        lines.append(f"\n{'='*40}\n")
        lines.append("Mask Statistics:\n")
        
        total_pixels = 0
        for class_name, mask in class_masks.items():
            pixels = np.sum(mask > 0)
            total_pixels += pixels
            if pixels > 0:
                percentage = 100 * pixels / mask.size
                lines.append(f"  {class_name}: {pixels} pixels ({percentage:.1f}%)\n")
        
        if total_pixels > 0:
            total_percentage = 100 * total_pixels / (info.get('image_height', 1) * info.get('image_width', 1))
            lines.append(f"\nTotal: {total_pixels} pixels ({total_percentage:.1f}%)\n")
        
        lines.append(f"\n{'='*40}\n")
        lines.append("Color Legend:\n")
        
        # Get class names from config (use class names, not prompts)
        class_names = info.get('class_names', info.get('prompts', []))
        class_colors_bgr = info.get('class_colors', {})
        
        # Color name mapping (approximate)
        def get_color_name(bgr_color):
            """Get approximate color name from BGR values."""
            if not isinstance(bgr_color, list) or len(bgr_color) != 3:
                return "Custom"
            b, g, r = bgr_color
            # Simple color detection
            if r > 200 and g < 100 and b < 100:
                return "Red"
            elif g > 200 and r < 100 and b < 100:
                return "Green"
            elif b > 200 and r < 100 and g < 100:
                return "Blue"
            elif r > 200 and g > 200 and b < 100:
                return "Yellow"
            elif r > 200 and b > 200 and g < 100:
                return "Magenta"
            elif g > 200 and b > 200 and r < 100:
                return "Cyan"
            else:
                return "Custom"
        
        for class_name in class_names:
            color_bgr = class_colors_bgr.get(class_name, None)
            if color_bgr:
                color_name = get_color_name(color_bgr)
            else:
                color_name = "Default"
            pixels = np.sum(class_masks.get(class_name, np.array([])) > 0)
            lines.append(f"  {color_name}: {class_name}\n")
        
        lines.append(f"\n{'='*40}\n")
        lines.append("Controls:\n")
        lines.append("  Accept: Save image and mask\n")
        lines.append("  Discard: Remove image\n")
        lines.append("  Reprocess: Regenerate masks\n")
        lines.append("  Quit: Exit application\n")
        
        self.info_text.insert(1.0, ''.join(lines))
        self.info_text.config(state=tk.DISABLED)
    
    def update_status(self, message: str):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()
    
    def destroy(self):
        """Destroy the GUI window."""
        self.root.destroy()
