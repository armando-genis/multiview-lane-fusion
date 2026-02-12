"""Simple GUI for ontology editing."""

import tkinter as tk
from tkinter import ttk, messagebox
import yaml
from pathlib import Path
from typing import Callable, Optional


class OntologyEditor:
    """Simple GUI for editing ontology YAML."""
    
    def __init__(self, config_path: Path, on_apply: Optional[Callable] = None):
        self.config_path = config_path
        self.on_apply = on_apply
        
        self.root = tk.Tk()
        self.root.title("Ontology Editor")
        self.root.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Label
        ttk.Label(main_frame, text="Edit Ontology Configuration:").grid(row=0, column=0, sticky=(tk.W), pady=(0, 5))
        
        # Text editor
        self.text_editor = tk.Text(main_frame, width=70, height=25, wrap=tk.WORD, font=('Courier', 10))
        self.text_editor.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.text_editor.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.text_editor.configure(yscrollcommand=scrollbar.set)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Save & Apply", command=self._save_and_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reload", command=self._load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.root.destroy).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Load initial config
        self._load_config()
    
    def _load_config(self):
        """Load config from file."""
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
            
            self.text_editor.delete(1.0, tk.END)
            self.text_editor.insert(1.0, content)
            self.status_var.set(f"Loaded: {self.config_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            self.status_var.set("Error loading config")
    
    def _save_and_apply(self):
        """Save config and notify."""
        try:
            # Get text
            content = self.text_editor.get(1.0, tk.END).strip()
            
            # Validate YAML
            yaml.safe_load(content)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                f.write(content)
            
            self.status_var.set("Saved successfully")
            messagebox.showinfo("Success", "Ontology saved! Press 'r' in terminal to reprocess.")
            
            # Call callback if provided
            if self.on_apply:
                self.on_apply()
                
        except yaml.YAMLError as e:
            messagebox.showerror("YAML Error", f"Invalid YAML syntax:\n{e}")
            self.status_var.set("Invalid YAML")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
            self.status_var.set("Error saving")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()
