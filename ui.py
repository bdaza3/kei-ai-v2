"""Tkinter popup overlay UI."""

from __future__ import annotations

import logging
import os
import tkinter as tk
from typing import Optional
from recorder import Recorder


class PopupOverlay:
    def __init__(self, width: int = 360, height: int = 140) -> None:
        self.width = width
        self.height = height
        self._root: Optional[tk.Tk] = None
        # _persistent_popup: stays visible until `close()` called
        self._persistent_popup: Optional[tk.Toplevel] = None
        self._persistent_label: Optional[tk.Label] = None
        # _temp_popup: used for transient messages that auto-close
        self._temp_popup: Optional[tk.Toplevel] = None
        # hold references to PhotoImage objects to avoid GC
        self._persistent_image: Optional[tk.PhotoImage] = None
        self._temp_image: Optional[tk.PhotoImage] = None

    def _ensure_root(self) -> tk.Tk:
        if self._root is None:
            self._root = tk.Tk()
            self._root.withdraw()
        return self._root

    def show_dialogue(self, text: str, duration_ms: int = 3500, image_path: Optional[str] = None) -> None:
        """Show a transient dialogue that auto-closes after `duration_ms` milliseconds.

        Transient dialogues are shown in a separate temporary popup so a persistent
        overlay (if created) remains visible.
        """
        root = self._ensure_root()

        # Close existing temp popup if present
        if self._temp_popup is not None:
            try:
                self._temp_popup.destroy()
            except Exception:
                pass
            self._temp_popup = None

        popup = tk.Toplevel(root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)

        # Try to load and (if possible) resize the image to the overlay width.
        img = self._load_image(image_path, target_width=self.width)

        # Calculate popup height: image height (if present) + text box height
        img_h = img.height() if img is not None else 0
        total_h = img_h + self.height

        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = max(0, screen_w - self.width - 24)
        y = max(0, screen_h - total_h - 48)
        popup.geometry(f"{self.width}x{total_h}+{x}+{y}")

        # If we have an image, place it above the dark message box and make
        # it the same width as the popup. Otherwise, just show the message box.
        if img is not None:
            img_label = tk.Label(popup, bg="#000000")
            img_label.pack(fill="x")
            img_label.configure(image=img)
            popup._img = img

        # dark message box
        box = tk.Frame(popup, bg="#1b1e24")
        box.pack(fill="both", expand=False)
        box.pack_propagate(False)
        box.configure(height=self.height)

        inner = tk.Frame(box, bg="#1b1e24")
        inner.pack(fill="both", expand=True, padx=12, pady=10)

        message = tk.Label(
            inner,
            text=text,
            fg="#e7e7e7",
            bg="#1b1e24",
            justify="left",
            wraplength=self.width - 24,
            font=("Segoe UI", 10),
        )
        message.pack(fill="both", expand=True)

        # schedule close for the temp popup
        popup.after(duration_ms, lambda: self._close_temp_popup(popup))
        self._temp_popup = popup
        self.process_events()

    def _load_image(self, image_path: Optional[str], target_width: Optional[int] = None) -> Optional[tk.PhotoImage]:
        if not image_path:
            return None
        try:
            # allow relative paths from repo root
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            if not os.path.exists(image_path):
                logging.warning("Avatar image not found: %s", image_path)
                return None
            
            img = tk.PhotoImage(file=image_path)
            if target_width and img.width() > target_width:
                factor = max(1, int(round(img.width() / target_width)))
                try:
                    img = img.subsample(factor, factor)
                except Exception:
                    pass
            return img
        
        except Exception:
            logging.exception("Failed to load avatar image: %s", image_path)
            return None

    def show_persistent(self, text: str, image_path: Optional[str] = None) -> None:
        """Show or update a persistent overlay that remains until `close()` is called."""
        root = self._ensure_root()

        if self._persistent_popup is not None:
            # update label text
            if self._persistent_label is not None:
                self._persistent_label.configure(text=text)
            self.process_events()
            return

        popup = tk.Toplevel(root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)

        img = self._load_image(image_path, target_width=self.width)
        img_h = img.height() if img is not None else 0
        total_h = img_h + self.height

        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = max(0, screen_w - self.width - 24)
        y = max(0, screen_h - total_h - 48)
        popup.geometry(f"{self.width}x{total_h}+{x}+{y}")

        if img is not None:
            img_label = tk.Label(popup, bg="#000000")
            img_label.pack(fill="x")
            img_label.configure(image=img)
            self._persistent_image = img

        box = tk.Frame(popup, bg="#1b1e24")
        box.pack(fill="both", expand=False)
        box.pack_propagate(False)
        box.configure(height=self.height)

        inner = tk.Frame(box, bg="#1b1e24")
        inner.pack(fill="both", expand=True, padx=12, pady=10)

        message = tk.Label(
            inner,
            text=text,
            fg="#e7e7e7",
            bg="#1b1e24",
            justify="left",
            wraplength=self.width - 24,
            font=("Segoe UI", 10),
        )
        message.pack(fill="both", expand=True)

        self._persistent_popup = popup
        self._persistent_label = message
        self.process_events()

    def _close_temp_popup(self, popup: Optional[tk.Toplevel] = None) -> None:
        # Close a specific temp popup (if provided), otherwise close stored temp
        if popup is not None:
            try:
                popup.destroy()
            except Exception:
                pass
            if self._temp_popup is popup:
                self._temp_popup = None
            return

        if self._temp_popup is not None:
            try:
                self._temp_popup.destroy()
            except Exception:
                pass
            self._temp_popup = None

    def process_events(self) -> None:
        if self._root is not None:
            self._root.update_idletasks()
            self._root.update()

    def close(self) -> None:
        # close both persistent and temp popups and root
        self._close_temp_popup()
        if self._persistent_popup is not None:
            try:
                self._persistent_popup.destroy()
            except Exception:
                pass
            self._persistent_popup = None
            self._persistent_label = None

        if self._root is not None:
            self._root.destroy()
            self._root = None
