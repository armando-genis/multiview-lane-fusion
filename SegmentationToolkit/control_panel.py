"""Control panel GUI for Rerun labeling."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict


class ControlPanel:
    """Simple control panel with buttons and prompt editor."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Labeling Controls")
        self.root.geometry("400x300")
        
        # Callbacks
        self.on_accept: Optional[Callable] = None
        self.on_discard: Optional[Callable] = None
        self.on_reprocess: Optional[Callable] = None
        self.on_edit_ontology: Optional[Callable] = None
        self.on_quit: Optional[Callable] = None
        
        self._setup_ui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """Setup the control panel UI."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Current status
        ttk.Label(main_frame, text="Current Image Status:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W
        )
        
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)
        
        # Action buttons
        ttk.Label(main_frame, text="Actions:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=2, column=0, columnspan=2, pady=(0, 5), sticky=tk.W
        )
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.accept_btn = ttk.Button(button_frame, text="Accept (N)", command=self._accept, width=15)
        self.accept_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.discard_btn = ttk.Button(button_frame, text="Discard (S)", command=self._discard, width=15)
        self.discard_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.reprocess_btn = ttk.Button(button_frame, text="Reprocess (R)", command=self._reprocess, width=15)
        self.reprocess_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.edit_btn = ttk.Button(button_frame, text="Edit Ontology (E)", command=self._edit_ontology, width=15)
        self.edit_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Quit button
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        self.quit_btn = ttk.Button(main_frame, text="Quit (Q)", command=self._quit, width=20)
        self.quit_btn.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Keyboard shortcuts
        self.root.bind('<n>', lambda e: self._accept())
        self.root.bind('<N>', lambda e: self._accept())
        self.root.bind('<s>', lambda e: self._discard())
        self.root.bind('<S>', lambda e: self._discard())
        self.root.bind('<r>', lambda e: self._reprocess())
        self.root.bind('<R>', lambda e: self._reprocess())
        self.root.bind('<e>', lambda e: self._edit_ontology())
        self.root.bind('<E>', lambda e: self._edit_ontology())
        self.root.bind('<q>', lambda e: self._quit())
        self.root.bind('<Q>', lambda e: self._quit())
    
    def _accept(self):
        if self.on_accept:
            self.on_accept()
    
    def _discard(self):
        if self.on_discard:
            self.on_discard()
    
    def _reprocess(self):
        if self.on_reprocess:
            self.on_reprocess()
    
    def _edit_ontology(self):
        if self.on_edit_ontology:
            self.on_edit_ontology()
    
    def _quit(self):
        if self.on_quit:
            self.on_quit()
    
    def _on_close(self):
        """Handle window close button."""
        self._quit()
    
    def update_status(self, status: str, color: str = "blue"):
        """Update the status label."""
        self.status_label.config(text=status, foreground=color)
    
    def set_buttons_enabled(self, enabled: bool):
        """Enable or disable action buttons."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.accept_btn.config(state=state)
        self.discard_btn.config(state=state)
        self.reprocess_btn.config(state=state)
    
    def run(self):
        """Run the control panel (blocking)."""
        self.root.mainloop()
    
    def update(self):
        """Update the GUI (non-blocking)."""
        self.root.update()
    
    def destroy(self):
        """Destroy the control panel."""
        try:
            self.root.destroy()
        except:
            pass
