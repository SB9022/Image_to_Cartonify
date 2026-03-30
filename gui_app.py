# gui_app.py
"""
Cartoonify Studio - Material 3 Tkinter GUI for AI-powered image cartoonification.
This is the main application file. It depends on 'image_effects.py'.
VERSION: 2.0 - Implemented a scrollable left-side panel for a fluid UI.
Author: (Your Name)
Date: 2025-09-24
"""

import os
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import cv2
import numpy as np
import logging
import time
import math
import copy
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageSequence

# --- Core application dependency ---
from Image_Effects import ImageEffects, ensure_odd

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# Material 3 Design System
class Material3Theme:
    # ... (This class is unchanged)
    # Material 3 Color Tokens
    COLORS = {
    # Primary colors (Lavender Fields palette)
    'primary': '#9C88D4',
    'on_primary': '#FFFFFF',
    'primary_container': '#7B68A8',
    'on_primary_container': '#F3EFFF',

    # Secondary colors
    'secondary': '#B8A5D6',
    'on_secondary': '#2D1B4E',
    'secondary_container': '#9680B8',
    'on_secondary_container': '#F8F4FF',

    # Tertiary colors
    'tertiary': '#E8B4A8',
    'on_tertiary': '#4A2417',
    'tertiary_container': '#C99489',
    'on_tertiary_container': '#FFF5F3',

    # Error colors
    'error': '#F28B82',
    'on_error': '#FFFFFF',
    'error_container': '#D14A3F',
    'on_error_container': '#FFEEED',

    # Surface colors
    'surface': '#1E1A2E',
    'on_surface': '#EDE8F5',
    'surface_variant': '#2B2640',
    'on_surface_variant': '#D4C9E8',
    'surface_container_lowest': '#15111F',
    'surface_container_low': '#1A1628',
    'surface_container': '#241F35',
    'surface_container_high': '#2F2A42',
    'surface_container_highest': '#3A344F',

    # Background colors
    'background': "#181425",
    'on_background': '#E8E0F5',

    # Outline colors
    'outline': '#A899C9',
    'outline_variant': '#4D3F7A',

    # Success (custom)
    'success': '#7CB982',
    'on_success': '#FFFFFF',
    'success_container': '#5A9865',
    'on_success_container': '#E8F5EA',

    # Warning (custom)
    'warning': '#F4BF75',
    'on_warning': '#3D2600',
    'warning_container': '#D19E55',
    'on_warning_container': '#FFF8ED',
    }
    
    # Typography Scale
    TYPOGRAPHY = {
        'display_large': ('Segoe UI', 57, 'normal'),
        'display_medium': ('Segoe UI', 45, 'normal'),
        'display_small': ('Segoe UI', 36, 'normal'),
        'headline_large': ('Segoe UI', 32, 'normal'),
        'headline_medium': ('Segoe UI', 28, 'normal'),
        'headline_small': ('Segoe UI', 24, 'normal'),
        'title_large': ('Segoe UI', 22, 'bold'),
        'title_medium': ('Segoe UI', 16, 'bold'),
        'title_small': ('Segoe UI', 14, 'bold'),
        'label_large': ('Segoe UI', 14, 'bold'),
        'label_medium': ('Segoe UI', 12, 'bold'),
        'label_small': ('Segoe UI', 11, 'bold'),
        'body_large': ('Segoe UI', 16, 'normal'),
        'body_medium': ('Segoe UI', 14, 'normal'),
        'body_small': ('Segoe UI', 12, 'normal'),
    }
    
    # Elevation levels
    ELEVATION = {
        'level_0': {'relief': 'flat', 'borderwidth': 0},
        'level_1': {'relief': 'solid', 'borderwidth': 1},
        'level_2': {'relief': 'raised', 'borderwidth': 1},
        'level_3': {'relief': 'raised', 'borderwidth': 2},
        'level_4': {'relief': 'raised', 'borderwidth': 2},
        'level_5': {'relief': 'raised', 'borderwidth': 3},
    }

# Enhanced constants for better UI scaling
THUMBNAIL_PREVIEW_SIZE = (380, 380)   
LIVE_PREVIEW_CROP = (250, 250)        

# --- Helper Functions for GUI and Data Handling ---
def pil_from_cv_rgb(cv_img):
    return Image.fromarray(cv_img.astype('uint8'))

def cv_rgb_from_pil(pil_img):
    return np.array(pil_img.convert('RGB'))

def safe_int(x, default=0):
    try:
        return int(x)
    except (ValueError, TypeError):
        return default

def safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _hex_to_rgb(hex_color):
    """Convert #RRGGBB string to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Expected color in #RRGGBB format")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb_tuple):
    """Convert an (r, g, b) tuple to #RRGGBB string."""
    return '#{:02X}{:02X}{:02X}'.format(*[max(0, min(255, int(round(c)))) for c in rgb_tuple])


def _blend_colors(color_a, color_b, factor):
    """Blend two hex colors together by a factor between 0-1."""
    factor = max(0.0, min(1.0, factor))
    ra, ga, ba = _hex_to_rgb(color_a)
    rb, gb, bb = _hex_to_rgb(color_b)
    blended = (
        ra + (rb - ra) * factor,
        ga + (gb - ga) * factor,
        ba + (bb - ba) * factor,
    )
    return _rgb_to_hex(blended)


def _darken_color(color, factor):
    """Darken a color by mixing it with black."""
    return _blend_colors(color, '#000000', factor)


class _ToolTip:
    """Simple tooltip helper bound to a Tk widget."""

    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<FocusOut>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            background="#26223A",
            foreground="#F4F0FF",
            relief=tk.SOLID,
            borderwidth=1,
            justify=tk.LEFT,
            padx=8,
            pady=4,
            font=("Segoe UI", 9),
            wraplength=260,
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

class CartoonifyStudio:
    def __init__(self, root):
        if DND_AVAILABLE and not isinstance(root, TkinterDnD.Tk):
            try:
                self.root = TkinterDnD.Tk()
            except Exception:
                self.root = root
        else:
            self.root = root

        self.theme = Material3Theme()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(1300, int(screen_width * 0.85))
        window_height = max(850, int(screen_height * 0.85))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.title("Cartoonify Studio")
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(background=self.theme.COLORS['background'])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(1100, 750)

        # Application state
        self.original_path = None
        self.original_cv_image = None   
        self.cartoon_cv_image = None    
        self.previous_cartoon_image = None
        self.stop_webcam_flag = threading.Event()
        self.preview_debounce_job = None

        self.defaults = {
            'edge_enhance': 2,
            'color_levels': 6,
            'smooth': 7,
            'contrast': 1.1,
            'cartoon_saturation': 1.05,
            'cartoon_value': 1.02,
            'cartoon_gamma': 1.0,
            'cartoon_local_contrast': 0.0,
            'line_thickness': 3,
            'canny_low': 50,
            'canny_high': 150,
            'bilateral_d': 11,
            'advanced_saturation': 1.2,
            'advanced_value': 1.0,
            'advanced_gamma': 1.0,
            'advanced_local_contrast': 0.0,
            'adaptive_mix': 0.45,
            'sharpen': 0.5,
            'style': 'Cartoon',
            'face_focus': False
        }

        self.spinner_frames = []
        self.spinner_running = False
        self._load_spinner()

        # Animated UI state
        self._animated_styles = []
        self._glow_phase = 0.0
        self._scroll_hint_phase = 0.0
        self.scroll_hint_label = None
        self._preview_progress_active = False
        self.preview_status_label = None
        self.preview_progress = None
        self._animation_job = None
        self._scroll_hint_job = None
        self._entrance_widgets = []
        self._entrance_phase = 0.0
        self._progress_animation_job = None
        self._progress_target = 0.0
        self._progress_current = 0.0
        self._preview_executor = ThreadPoolExecutor(max_workers=1)
        self._preview_cancel_event = threading.Event()
        self._tooltips = []

        self._setup_material3_styles()
        self._create_material3_layout()
        self._start_button_animation_loop()
        self._start_scroll_hint_animation()
        self._start_entrance_animations()
        self._bind_shortcuts()

        try:
            face_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            self.face_cascade = cv2.CascadeClassifier(face_xml)
        except Exception:
            self.face_cascade = None

    # ... (Most of the CartoonifyStudio methods are unchanged) ...
    def _on_close(self):
        self.stop_webcam_flag.set()
        if self._preview_cancel_event:
            self._preview_cancel_event.set()
        if self.preview_debounce_job:
            try:
                self.root.after_cancel(self.preview_debounce_job)
            except tk.TclError:
                pass
            self.preview_debounce_job = None
        if self._progress_animation_job:
            try:
                self.root.after_cancel(self._progress_animation_job)
            except tk.TclError:
                pass
            self._progress_animation_job = None
        try:
            if getattr(self, 'progress_bar', None) is not None:
                self.progress_bar.stop()
        except tk.TclError:
            pass
        if getattr(self, 'preview_progress', None) is not None:
            try:
                self.preview_progress.stop()
            except tk.TclError:
                pass
        self._preview_progress_active = False
        if self._preview_executor:
            self._preview_executor.shutdown(wait=False, cancel_futures=True)
        self.root.quit()
        self.root.destroy()

    def _load_spinner(self):
        spinner_path = os.path.join(os.path.dirname(__file__), "spinner.gif")
        if not os.path.exists(spinner_path): return
        try:
            spinner = Image.open(spinner_path)
            for frame in ImageSequence.Iterator(spinner):
                self.spinner_frames.append(ImageTk.PhotoImage(frame.convert('RGBA')))
        except Exception as e:
            logging.warning(f"Could not load spinner GIF: {e}")
            self.spinner_frames = []

    def _attach_tooltip(self, widget, text):
        if not text:
            return
        tooltip = _ToolTip(widget, text)
        self._tooltips.append(tooltip)

    def _setup_material3_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Filled Button (Primary)
        self.style.configure('M3.Filled.TButton', font=self.theme.TYPOGRAPHY['label_large'], padding=(24, 12), relief='flat', borderwidth=0, focuscolor='none')
        self.style.map('M3.Filled.TButton', background=[('active', self.theme.COLORS['primary_container']), ('pressed', self.theme.COLORS['primary']), ('!active', self.theme.COLORS['primary'])], foreground=[('active', self.theme.COLORS['on_primary_container']), ('pressed', self.theme.COLORS['on_primary']), ('!active', self.theme.COLORS['on_primary'])])
        
        # Filled Tonal Button (Secondary)
        self.style.configure('M3.FilledTonal.TButton', font=self.theme.TYPOGRAPHY['label_large'], padding=(24, 12), relief='flat', borderwidth=0, focuscolor='none')
        self.style.map('M3.FilledTonal.TButton', background=[('active', self.theme.COLORS['secondary_container']), ('!active', self.theme.COLORS['secondary_container'])], foreground=[('active', self.theme.COLORS['on_secondary_container']), ('!active', self.theme.COLORS['on_secondary_container'])])
        
        # Outlined Button
        self.style.configure('M3.Outlined.TButton', font=self.theme.TYPOGRAPHY['label_large'], padding=(24, 12), relief='solid', borderwidth=1, focuscolor='none')
        self.style.map('M3.Outlined.TButton', background=[('active', self.theme.COLORS['surface']), ('!active', self.theme.COLORS['surface'])], foreground=[('active', self.theme.COLORS['primary']), ('!active', self.theme.COLORS['primary'])], bordercolor=[('active', self.theme.COLORS['outline']), ('!active', self.theme.COLORS['outline'])])
        
        # Text Button
        self.style.configure('M3.Text.TButton', font=self.theme.TYPOGRAPHY['label_large'], padding=(12, 12), relief='flat', borderwidth=0, focuscolor='none')
        self.style.map('M3.Text.TButton', background=[('active', self.theme.COLORS['surface_container_highest']), ('!active', self.theme.COLORS['surface'])], foreground=[('active', self.theme.COLORS['primary']), ('!active', self.theme.COLORS['primary'])])
        
        # Success Button
        self.style.configure('M3.Success.TButton', font=self.theme.TYPOGRAPHY['label_large'], padding=(24, 12), relief='flat', borderwidth=0, focuscolor='none')
        self.style.map('M3.Success.TButton', background=[('active', self.theme.COLORS['success_container']), ('!active', self.theme.COLORS['success'])], foreground=[('active', self.theme.COLORS['on_success_container']), ('!active', self.theme.COLORS['on_success'])])
        
        # Other Styles (Labels, Frames, etc.) are unchanged
        self.style.configure('M3.HeadlineLarge.TLabel', font=self.theme.TYPOGRAPHY['headline_large'], foreground=self.theme.COLORS['on_background'], background=self.theme.COLORS['background'])
        self.style.configure('M3.BodyLarge.TLabel', font=self.theme.TYPOGRAPHY['body_large'], foreground=self.theme.COLORS['on_surface_variant'], background=self.theme.COLORS['surface'])
        self.style.configure('M3.TitleLarge.TLabel', font=self.theme.TYPOGRAPHY['title_large'], foreground=self.theme.COLORS['on_surface'], background=self.theme.COLORS['surface'])
        self.style.configure('M3.TitleMedium.TLabel', font=self.theme.TYPOGRAPHY['title_medium'], foreground=self.theme.COLORS['on_surface'], background=self.theme.COLORS['surface'])
        self.style.configure('M3.BodyMedium.TLabel', font=self.theme.TYPOGRAPHY['body_medium'], foreground=self.theme.COLORS['on_surface_variant'], background=self.theme.COLORS['surface'])
        self.style.configure('M3.LabelLarge.TLabel', font=self.theme.TYPOGRAPHY['label_large'], foreground=self.theme.COLORS['on_surface'], background=self.theme.COLORS['surface'])
        self.style.configure('M3.LabelSmall.TLabel', font=self.theme.TYPOGRAPHY['label_small'], foreground=self.theme.COLORS['on_surface_variant'], background=self.theme.COLORS['surface_container'])
        self.style.configure('M3.Surface.TFrame', background=self.theme.COLORS['surface'])
        self.style.configure('M3.SurfaceContainer.TFrame', background=self.theme.COLORS['surface_container'], relief='solid', borderwidth=1, bordercolor=self.theme.COLORS['outline_variant'])
        self.style.configure('M3.SurfaceContainerHigh.TFrame', background=self.theme.COLORS['surface_container_high'], relief='raised', borderwidth=1)
        self.style.configure('M3.PrimaryContainer.TFrame', background=self.theme.COLORS['primary_container'])

        base_progress_layout = copy.deepcopy(self.style.layout('Horizontal.TProgressbar'))
        self.style.layout('M3.Horizontal.TProgressbar', base_progress_layout)

        accent_progress_layout = copy.deepcopy(self.style.layout('Horizontal.TProgressbar'))
        self.style.layout('M3Accent.Horizontal.TProgressbar', accent_progress_layout)

        base_progress_options = {
            'thickness': 6,
            'troughcolor': self.theme.COLORS['surface_container_highest'],
            'background': self.theme.COLORS['primary'],
            'borderwidth': 0,
            'relief': 'flat'
        }
        self.style.configure('M3.Horizontal.TProgressbar', **base_progress_options)

        accent_progress_options = {
            'thickness': 4,
            'troughcolor': self.theme.COLORS['surface_container_high'],
            'background': self.theme.COLORS['tertiary'],
            'borderwidth': 0,
            'relief': 'flat'
        }
        self.style.configure('M3Accent.Horizontal.TProgressbar', **accent_progress_options)
        self.style.configure('M3.TCombobox', fieldbackground=self.theme.COLORS['surface_container'], background=self.theme.COLORS['surface_container'], foreground=self.theme.COLORS['on_surface'], arrowcolor=self.theme.COLORS['on_surface_variant'], borderwidth=1, bordercolor=self.theme.COLORS['outline'], font=self.theme.TYPOGRAPHY['body_large'])
        self.style.configure('M3.Horizontal.TScale', background=self.theme.COLORS['surface_container'], troughcolor=self.theme.COLORS['surface_container_highest'], slidercolor=self.theme.COLORS['primary'], borderwidth=0, relief='flat')
        self.style.configure('M3.TCheckbutton', font=self.theme.TYPOGRAPHY['body_large'], foreground=self.theme.COLORS['on_surface'], background=self.theme.COLORS['surface'], focuscolor='none')

    def _register_animated_button(self, widget, base_color, accent_color, text_color=None, padding=(24, 12)):
        """Create a glowing animated style for a given ttk button widget."""
        if getattr(self, 'style', None) is None or widget is None:
            return

        text_color = text_color or self.theme.COLORS['on_primary']
        style_name = f"M3.Glow.{len(self._animated_styles)}.TButton"

        self.style.configure(
            style_name,
            font=self.theme.TYPOGRAPHY['label_large'],
            padding=padding,
            relief='flat',
            borderwidth=2,
            background=base_color,
            foreground=text_color,
            focuscolor=self.theme.COLORS['surface'],
            focusthickness=0
        )
        self.style.map(
            style_name,
            background=[('pressed', _darken_color(base_color, 0.25)), ('disabled', _blend_colors(base_color, '#BBBBBB', 0.5))],
            foreground=[('disabled', _blend_colors(text_color, '#777777', 0.4))],
            bordercolor=[('disabled', _blend_colors(base_color, '#888888', 0.5))]
        )

        widget.configure(style=style_name, cursor='hand2')

        info = {
            'style': style_name,
            'base': base_color,
            'accent': accent_color,
            'text': text_color,
            'hovered': False,
            'hover_transition': 0.0,
            'phase_offset': len(self._animated_styles) * 0.6,
        }
        self._animated_styles.append(info)

        widget.bind("<Enter>", lambda _e, data=info: self._set_button_hover(data, True))
        widget.bind("<Leave>", lambda _e, data=info: self._set_button_hover(data, False))
        widget.bind("<FocusIn>", lambda _e, data=info: self._set_button_hover(data, True))
        widget.bind("<FocusOut>", lambda _e, data=info: self._set_button_hover(data, False))
        
        # Add tactile press feedback
        widget.bind("<ButtonPress-1>", lambda _e, w=widget: self._on_button_press(w))
        widget.bind("<ButtonRelease-1>", lambda _e, w=widget: self._on_button_release(w))

        if self._animation_job is None:
            self._start_button_animation_loop()

    def _on_button_press(self, widget):
        """Visual feedback on button press."""
        try:
            if widget.winfo_exists():
                widget.configure(cursor='hand2')
        except (tk.TclError, AttributeError):
            pass

    def _on_button_release(self, widget):
        """Reset visual feedback on button release."""
        try:
            if widget.winfo_exists():
                widget.configure(cursor='hand2')
        except (tk.TclError, AttributeError):
            pass

    def _set_button_hover(self, info, hovered):
        info['hovered'] = hovered

    def _start_button_animation_loop(self):
        if self._animation_job is not None:
            self.root.after_cancel(self._animation_job)
        self._animate_button_styles()

    def _animate_button_styles(self):
        if not self._animated_styles or not getattr(self, 'style', None):
            return

        self._glow_phase = (self._glow_phase + 0.22) % (2 * math.pi)
        
        for info in self._animated_styles:
            style_name = info['style']
            base = info['base']
            accent = info['accent']
            text = info['text']
            phase = self._glow_phase + info['phase_offset']
            
            # Smooth hover transition
            target_hover = 1.0 if info['hovered'] else 0.0
            info['hover_transition'] += (target_hover - info['hover_transition']) * 0.15
            
            # Calculate pulsing effect
            pulse = (math.sin(phase) + 1) * 0.5
            
            # Combine pulse with hover
            hover_boost = info['hover_transition'] * 0.5
            blend_factor = 0.15 + pulse * 0.35 + hover_boost
            border_factor = 0.25 + pulse * 0.45 + hover_boost * 0.6
            
            try:
                self.style.configure(
                    style_name,
                    background=_blend_colors(base, accent, blend_factor),
                    bordercolor=_blend_colors(accent, '#FFFFFF', border_factor * 0.7),
                    foreground=text
                )
            except tk.TclError:
                continue

        self._animation_job = self.root.after(50, self._animate_button_styles)

    def _start_scroll_hint_animation(self):
        if self.scroll_hint_label is None or not isinstance(self.scroll_hint_label, ttk.Label):
            return
        if self._scroll_hint_job is not None:
            self.root.after_cancel(self._scroll_hint_job)
        self._animate_scroll_hint()

    def _animate_scroll_hint(self):
        if self.scroll_hint_label is None or not self.scroll_hint_label.winfo_exists():
            return
        self._scroll_hint_phase = (self._scroll_hint_phase + 0.15) % (2 * math.pi)
        
        # Create smooth pulsing effect
        pulse = (math.sin(self._scroll_hint_phase) + 1) * 0.5
        
        # Smooth easing
        eased_pulse = pulse * pulse * (3 - 2 * pulse)  # smoothstep
        
        color = _blend_colors(
            self.theme.COLORS['on_surface_variant'], 
            self.theme.COLORS['primary'], 
            0.3 + eased_pulse * 0.6
        )
        
        try:
            self.scroll_hint_label.configure(foreground=color)
        except tk.TclError:
            return
        self._scroll_hint_job = self.root.after(60, self._animate_scroll_hint)

    def _start_preview_progress(self):
        if self.preview_progress is None:
            return
        if not self._preview_progress_active:
            self.preview_progress.start(35)
            if self.preview_status_label is not None:
                self.preview_status_label.configure(text="Updating live preview…")
            self._preview_progress_active = True

    def _stop_preview_progress(self, success=True, message=None):
        if self.preview_progress is None:
            return
        if self._preview_progress_active:
            self.preview_progress.stop()
            self._preview_progress_active = False
            if self.preview_status_label is not None:
                status_text = message if message is not None else ("Live preview synced" if success else "Preview update failed")
                self.preview_status_label.configure(text=status_text)

    def _start_entrance_animations(self):
        """Animate widgets fading in on startup."""
        self._entrance_phase = 0.0
        self._animate_entrance()

    def _animate_entrance(self):
        """Progressive fade-in animation for UI elements."""
        if self._entrance_phase >= 1.0:
            return
        
        self._entrance_phase = min(1.0, self._entrance_phase + 0.08)
        
        # Apply easing function for smooth acceleration
        eased = 1 - math.pow(1 - self._entrance_phase, 3)
        
        # Animate registered widgets
        for widget_info in self._entrance_widgets:
            widget = widget_info['widget']
            delay = widget_info.get('delay', 0)
            
            if self._entrance_phase < delay:
                continue
                
            adjusted_phase = min(1.0, (self._entrance_phase - delay) / (1.0 - delay) if delay < 1.0 else 1.0)
            eased_adjusted = 1 - math.pow(1 - adjusted_phase, 3)
            
            try:
                if widget.winfo_exists():
                    # Fade in effect by adjusting widget state
                    if hasattr(widget, 'configure'):
                        current_state = widget.cget('state') if 'state' in widget.keys() else None
                        if current_state == 'disabled' and eased_adjusted >= 0.5:
                            widget.configure(state='normal')
            except (tk.TclError, AttributeError):
                continue
        
        if self._entrance_phase < 1.0:
            self.root.after(50, self._animate_entrance)

    def _register_entrance_animation(self, widget, delay=0.0):
        """Register a widget for entrance animation."""
        self._entrance_widgets.append({
            'widget': widget,
            'delay': delay
        })

    # ------------------- NEW HELPER METHODS FOR SCROLLING -------------------
    def _on_frame_configure(self, canvas):
        """Reset the scroll region to encompass the inner frame"""
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_mousewheel(self, event, canvas):
        """Handle mouse wheel scrolling"""
        # The delta is different on Windows/macOS vs Linux
        if event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")
    # -------------------------------------------------------------------------


    def _create_material3_header(self, container):
        """Create the Material 3 header with title and subtitle"""
        header_frame = ttk.Frame(container, style='M3.Surface.TFrame')
        header_frame.pack(fill='x', anchor='n')
        
        title_label = ttk.Label(header_frame, 
                              text="Cartoonify Studio", 
                              style='M3.HeadlineLarge.TLabel')
        title_label.pack(anchor='w')
        
        subtitle_label = ttk.Label(header_frame, 
                                 text="Transform your photos into stunning cartoons with AI-powered effects", 
                                 style='M3.BodyLarge.TLabel')
        subtitle_label.pack(anchor='w', pady=(4, 0))

    def _create_material3_layout(self):
        main = ttk.Frame(self.root, style='M3.Surface.TFrame')
        main.pack(fill='both', expand=True, padx=24, pady=24)

        self._create_material3_header(main)
        
        content_frame = ttk.Frame(main, style='M3.Surface.TFrame')
        content_frame.pack(fill='both', expand=True, pady=(24, 0))
        
        nav_rail = self._create_navigation_rail(content_frame)
        nav_rail.pack(side='left', fill='y', padx=(0, 24))
        
        main_content = self._create_main_content(content_frame)
        main_content.pack(side='right', fill='both', expand=True)
        
        self._create_bottom_app_bar(main)
        
        if DND_AVAILABLE:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self._on_drop)
            except Exception as e:
                logging.error(f"Drag-and-drop setup failed: {e}")

        self._photo_before = None
        self._photo_after = None

        all_sliders = (
            self.edge_scale, self.color_scale, self.smooth_scale, self.contrast_scale,
            self.line_thickness_scale, self.canny_low_scale, self.canny_high_scale,
            self.bilateral_d_scale, self.saturation_scale, self.sharpen_scale
        )
        for widget in (*all_sliders, self.style_combobox):
            widget.bind("<ButtonRelease-1>", self._on_control_change)
            widget.bind("<KeyRelease>", self._on_control_change)
        self.style_combobox.bind("<<ComboboxSelected>>", self._on_control_change)

    # ------------------- MAJOR CHANGE: SCROLLABLE NAVIGATION RAIL -------------------
    def _create_navigation_rail(self, parent):
        """Create Material 3 style navigation rail with a scrollbar."""
        # Main container for the rail and scrollbar
        rail_container = ttk.Frame(parent, style='M3.Surface.TFrame')

        # Canvas widget will act as a viewport for our scrollable content
        canvas = tk.Canvas(rail_container, bg=self.theme.COLORS['surface_container'], highlightthickness=0)
        
        # Scrollbar that controls the canvas view
        scrollbar = ttk.Scrollbar(rail_container, orient="vertical", command=canvas.yview)
        
        # This frame will contain all the controls and will be placed *inside* the canvas
        scrollable_frame = ttk.Frame(canvas, style='M3.SurfaceContainer.TFrame')

        # This binding is crucial: when the size of the content frame changes,
        # it updates the canvas's scrollable area.
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self._on_frame_configure(canvas)
        )

        # Place the content frame inside the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar into the main container
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Bind mouse wheel scrolling to the canvas and its contents ---
        # This makes scrolling intuitive
        mouse_scroll_handler = lambda event, c=canvas: self._on_mousewheel(event, c)
        canvas.bind_all("<MouseWheel>", mouse_scroll_handler) # Windows/macOS
        canvas.bind_all("<Button-4>", mouse_scroll_handler)   # Linux up
        canvas.bind_all("<Button-5>", mouse_scroll_handler)   # Linux down

        # --- NOW, ADD ALL THE SECTIONS TO THE 'scrollable_frame' INSTEAD OF THE OLD FRAME ---
        self._create_input_section_m3(scrollable_frame)
        self._create_style_section_m3(scrollable_frame)
        self._create_parameters_section_m3(scrollable_frame)
        self._create_actions_section_m3(scrollable_frame)

        self.scroll_hint_label = ttk.Label(scrollable_frame, text="⇅ Scroll for more controls", style='M3.LabelSmall.TLabel')
        self.scroll_hint_label.pack(anchor='center', pady=(12, 24))
        
        return rail_container
    # ----------------------------------------------------------------------------------

    # The rest of the creation methods are now simpler as they just pack into their parent
    def _create_input_section_m3(self, parent):
        section = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        section.pack(fill='x', pady=(16, 24), padx=16, expand=True)
        ttk.Label(section, text="Input", style='M3.TitleMedium.TLabel').pack(anchor='w', pady=(16, 16))

        browse_btn = ttk.Button(section, text="Browse Files", command=self.upload, style='M3.Filled.TButton')
        browse_btn.pack(fill='x', pady=(0, 12))
        self._register_animated_button(
            browse_btn,
            base_color=self.theme.COLORS['primary'],
            accent_color=self.theme.COLORS['primary_container'],
            text_color=self.theme.COLORS['on_primary']
        )

        batch_btn = ttk.Button(section, text="Batch Process", command=self.batch_select, style='M3.FilledTonal.TButton')
        batch_btn.pack(fill='x', pady=(0, 12))
        self._register_animated_button(
            batch_btn,
            base_color=self.theme.COLORS['secondary_container'],
            accent_color=self.theme.COLORS['secondary'],
            text_color=self.theme.COLORS['on_secondary_container']
        )

        webcam_btn = ttk.Button(section, text="Webcam Mode", command=self.start_webcam, style='M3.Outlined.TButton')
        webcam_btn.pack(fill='x', pady=(0, 16))
        webcam_btn.bind("<Enter>", lambda _e: webcam_btn.configure(cursor='hand2'))

    def _create_style_section_m3(self, parent):
        section = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        section.pack(fill='x', pady=(0, 24), padx=16, expand=True)
        ttk.Label(section, text="Style", style='M3.TitleMedium.TLabel').pack(anchor='w', pady=(16, 16))
        
        self.style_var = tk.StringVar(value=self.defaults['style'])
        self.style_combobox = ttk.Combobox(section, textvariable=self.style_var, values=["Cartoon", "Cartoon V2 (Advanced)", "Pencil Sketch", "Watercolor", "Comic Dots"], state='readonly', style='M3.TCombobox')
        self.style_combobox.pack(fill='x', pady=(0, 16))
        self.style_combobox.bind("<<ComboboxSelected>>", self._update_parameter_widgets)
        
        self.face_focus_var = tk.BooleanVar(value=self.defaults['face_focus'])
        self.face_check = ttk.Checkbutton(section, text="Emphasize faces", variable=self.face_focus_var, style='M3.TCheckbutton')
        self.face_check.pack(anchor='w', pady=(0, 16))

    def _create_parameters_section_m3(self, parent):
        self.params_section = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        self.params_section.pack(fill='x', pady=(0, 24), padx=16, expand=True)
        ttk.Label(self.params_section, text="Parameters", style='M3.TitleMedium.TLabel').pack(anchor='w', pady=(16, 16))

        self.standard_params_frame = ttk.Frame(self.params_section, style='M3.SurfaceContainer.TFrame')
        self.advanced_params_frame = ttk.Frame(self.params_section, style='M3.SurfaceContainer.TFrame')

        self.edge_scale = self._create_material3_slider(
            self.standard_params_frame,
            "Edge Enhancement",
            1,
            6,
            self.defaults['edge_enhance'],
            tooltip="Controls how strongly outlines are reinforced before blending. Higher values create thicker, bolder ink lines.")
        self.color_scale = self._create_material3_slider(
            self.standard_params_frame,
            "Color Levels",
            2,
            16,
            self.defaults['color_levels'],
            tooltip="Sets the number of color clusters used in quantization. Lower values give flatter, poster-like fills; higher values retain more gradients.")
        self.smooth_scale = self._create_material3_slider(
            self.standard_params_frame,
            "Smoothing",
            3,
            21,
            self.defaults['smooth'],
            tooltip="Base bilateral filter diameter. Automatically scales with resolution to avoid over-blur on large photos.")
        self.contrast_scale = self._create_material3_slider(
            self.standard_params_frame,
            "Contrast",
            0.6,
            2.0,
            self.defaults['contrast'],
            resolution=0.05,
            tooltip="Global contrast multiplier applied after color quantization. Try reducing slightly if highlights blow out.")
        self.cartoon_tone_frame = ttk.Frame(self.standard_params_frame, style='M3.SurfaceContainer.TFrame')
        self.cartoon_tone_frame.pack(fill='x', pady=(0, 16))
        self.cartoon_saturation_scale = self._create_material3_slider(
            self.cartoon_tone_frame,
            "Saturation Boost",
            0.8,
            1.6,
            self.defaults['cartoon_saturation'],
            resolution=0.05,
            tooltip="Adjusts how vivid the final colors appear. Values above 1.0 intensify chroma.")
        self.cartoon_value_scale = self._create_material3_slider(
            self.cartoon_tone_frame,
            "Brightness Boost",
            0.8,
            1.4,
            self.defaults['cartoon_value'],
            resolution=0.05,
            tooltip="Tweaks the value channel in HSV space to gently brighten or darken the cartoon result.")
        self.cartoon_gamma_scale = self._create_material3_slider(
            self.cartoon_tone_frame,
            "Tone Gamma",
            0.7,
            1.4,
            self.defaults['cartoon_gamma'],
            resolution=0.05,
            tooltip="Applies gamma correction for tone-mapping. Values below 1.0 brighten shadows; above 1.0 adds cinematic contrast.")
        self.cartoon_local_contrast_scale = self._create_material3_slider(
            self.cartoon_tone_frame,
            "Local Contrast",
            0.0,
            2.0,
            self.defaults['cartoon_local_contrast'],
            resolution=0.1,
            tooltip="Boost local detail using CLAHE. Small values subtly enhance textures; higher values can introduce grain.")

        self.line_thickness_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Line Thickness",
            0,
            9,
            self.defaults['line_thickness'],
            tooltip="Morphological closing size. Larger values merge gaps and yield heavier ink strokes.")
        self.canny_low_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Canny Low Threshold",
            10,
            150,
            self.defaults['canny_low'],
            tooltip="Base lower threshold for Canny edges. The algorithm blends this with an adaptive estimate from image statistics.")
        self.canny_high_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Canny High Threshold",
            50,
            250,
            self.defaults['canny_high'],
            tooltip="Base upper threshold for Canny edges. Controls sensitivity to fine line work.")
        self.bilateral_d_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Bilateral Filter Size",
            5,
            25,
            self.defaults['bilateral_d'],
            tooltip="Base bilateral diameter for advanced smoothing. Automatically scales with resolution to prevent over-blur.")
        self.saturation_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Saturation Boost",
            0.5,
            2.5,
            self.defaults['advanced_saturation'],
            resolution=0.1,
            tooltip="Chromatic boost applied after advanced processing. Keep near 1.0 for natural palettes.")
        self.value_boost_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Brightness Boost",
            0.7,
            1.4,
            self.defaults['advanced_value'],
            resolution=0.05,
            tooltip="Value channel multiplier for the advanced pipeline. Useful for lifting shadow detail.")
        self.tone_gamma_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Tone Gamma",
            0.7,
            1.4,
            self.defaults['advanced_gamma'],
            resolution=0.05,
            tooltip="Gamma remapping for the advanced cartoon effect. Lower values brighten mids; higher values deepen tones.")
        self.local_contrast_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Local Contrast",
            0.0,
            2.0,
            self.defaults['advanced_local_contrast'],
            resolution=0.1,
            tooltip="CLAHE-based micro-contrast enhancer. Adds punch to textures after smoothing.")
        self.adaptive_mix_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Adaptive Edge Mix",
            0.0,
            1.0,
            self.defaults['adaptive_mix'],
            resolution=0.05,
            tooltip="Blend factor between manual Canny thresholds and automatic thresholds derived from image statistics.")
        self.sharpen_scale = self._create_material3_slider(
            self.advanced_params_frame,
            "Sharpen Amount",
            0.0,
            1.5,
            self.defaults['sharpen'],
            resolution=0.1,
            tooltip="Strength of the unsharp mask applied at the end of the advanced pipeline.")

        live_preview_container = ttk.Frame(self.params_section, style='M3.SurfaceContainer.TFrame')
        live_preview_container.pack(fill='x', pady=(16,0))
        self.live_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(live_preview_container, text="Live preview", variable=self.live_preview_var, command=self._toggle_live_preview, style='M3.TCheckbutton').pack(anchor='w', pady=(0, 16))

        preview_status_frame = ttk.Frame(live_preview_container, style='M3.SurfaceContainer.TFrame')
        preview_status_frame.pack(fill='x', pady=(0, 16))
        self.preview_status_label = ttk.Label(preview_status_frame, text="Live preview ready", style='M3.BodyMedium.TLabel')
        self.preview_status_label.pack(anchor='w')
        self.preview_progress = ttk.Progressbar(preview_status_frame, mode='indeterminate', style='M3Accent.Horizontal.TProgressbar')
        self.preview_progress.pack(fill='x', pady=(6, 0))
        self.preview_progress.stop()
        
        self._update_parameter_widgets()
    
    # ... (All other methods like _create_actions_section_m3, _create_main_content, cartoonify_threaded, save, etc., remain unchanged) ...
    def _create_actions_section_m3(self, parent):
        section = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        section.pack(fill='x', pady=(0, 16), padx=16, expand=True)
        
        ttk.Label(section, text="Actions", style='M3.TitleMedium.TLabel').pack(anchor='w', pady=(16, 16))
        
        self.process_btn = ttk.Button(section, text="Cartoonify", command=self.cartoonify_threaded, style='M3.Filled.TButton', state='disabled')
        self.process_btn.pack(fill='x', pady=(0, 12))
        self._register_animated_button(
            self.process_btn,
            base_color=self.theme.COLORS['primary'],
            accent_color=self.theme.COLORS['primary_container'],
            text_color=self.theme.COLORS['on_primary']
        )
        
        actions_row1 = ttk.Frame(section, style='M3.SurfaceContainer.TFrame')
        actions_row1.pack(fill='x', pady=(0, 12))
        
        self.save_btn = ttk.Button(actions_row1, text="Save", command=self.save, style='M3.Success.TButton', state='disabled')
        self.save_btn.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self._register_animated_button(
            self.save_btn,
            base_color=self.theme.COLORS['success'],
            accent_color=self.theme.COLORS['success_container'],
            text_color=self.theme.COLORS['on_success']
        )
        
        self.compare_btn = ttk.Button(actions_row1, text="Compare", command=self.export_comparison, style='M3.FilledTonal.TButton', state='disabled')
        self.compare_btn.pack(side='right', fill='x', expand=True, padx=(6, 0))
        self._register_animated_button(
            self.compare_btn,
            base_color=self.theme.COLORS['secondary'],
            accent_color=self.theme.COLORS['secondary_container'],
            text_color=self.theme.COLORS['on_secondary_container']
        )
        
        actions_row2 = ttk.Frame(section, style='M3.SurfaceContainer.TFrame')
        actions_row2.pack(fill='x', pady=(0, 16))
        
        self.reset_btn = ttk.Button(actions_row2, text="Reset", command=self.reset_to_defaults, style='M3.Text.TButton')
        self.reset_btn.pack(side='left', fill='x', expand=True, padx=(0, 6))
        
        self.undo_btn = ttk.Button(actions_row2, text="Undo", command=self.undo, state='disabled', style='M3.Outlined.TButton')
        self.undo_btn.pack(side='right', fill='x', expand=True, padx=(6, 0))

    def _create_main_content(self, parent):
        content_container = ttk.Frame(parent, style='M3.Surface.TFrame')
        content_surface = ttk.Frame(content_container, style='M3.SurfaceContainer.TFrame')
        content_surface.pack(fill='both', expand=True, padx=8, pady=8)
        content_header = ttk.Frame(content_surface, style='M3.SurfaceContainer.TFrame')
        content_header.pack(fill='x', pady=(24, 16), padx=24)
        ttk.Label(content_header, text="Preview", style='M3.TitleLarge.TLabel').pack(anchor='w')
        labels_frame = ttk.Frame(content_surface, style='M3.SurfaceContainer.TFrame')
        labels_frame.pack(fill='x', pady=(0, 16), padx=24)
        ttk.Label(labels_frame, text="Original", style='M3.TitleMedium.TLabel').pack(side='left', expand=True)
        ttk.Label(labels_frame, text="Processed", style='M3.TitleMedium.TLabel').pack(side='right', expand=True)
        images_frame = ttk.Frame(content_surface, style='M3.SurfaceContainer.TFrame')
        images_frame.pack(fill='both', expand=True, padx=24, pady=(0, 24))
        before_container = ttk.Frame(images_frame, style='M3.SurfaceContainerHigh.TFrame')
        before_container.pack(side='left', fill='both', expand=True, padx=(0, 12))
        self.before_canvas = tk.Label(before_container, text="Drop an image here\nor click Browse Files\n\n📷", bg=self.theme.COLORS['surface_container_highest'], fg=self.theme.COLORS['on_surface_variant'], font=self.theme.TYPOGRAPHY['body_large'], anchor='center', relief='solid', borderwidth=2, bd=2)
        self.before_canvas.configure(highlightthickness=1, highlightcolor=self.theme.COLORS['outline_variant'], highlightbackground=self.theme.COLORS['outline_variant'])
        self.before_canvas.pack(fill='both', expand=True, padx=12, pady=12)
        after_container = ttk.Frame(images_frame, style='M3.SurfaceContainerHigh.TFrame')
        after_container.pack(side='right', fill='both', expand=True, padx=(12, 0))
        self.after_canvas = tk.Label(after_container, text="Processed image\nwill appear here\n\n🎨", bg=self.theme.COLORS['surface_container_highest'], fg=self.theme.COLORS['on_surface_variant'], font=self.theme.TYPOGRAPHY['body_large'], anchor='center', relief='solid', borderwidth=2, bd=2)
        self.after_canvas.configure(highlightthickness=1, highlightcolor=self.theme.COLORS['outline_variant'], highlightbackground=self.theme.COLORS['outline_variant'])
        self.after_canvas.pack(fill='both', expand=True, padx=12, pady=12)
        return content_container

    def _create_bottom_app_bar(self, parent):
        bottom_bar = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        bottom_bar.pack(fill='x', pady=(24, 0))
        self.status_frame = bottom_bar
        self.status_label = ttk.Label(bottom_bar, text="Ready • Select an image to begin", style='M3.BodyMedium.TLabel')
        self.status_label.pack(fill='x', padx=24, pady=16)
        self.progress_bar = ttk.Progressbar(bottom_bar, mode='indeterminate', style='M3.Horizontal.TProgressbar')
        # self.progress_bar.pack(fill='x', padx=24, pady=(0, 16)) # Packed on demand
        # self.progress_bar.pack_forget()

    def _create_material3_slider(self, parent, label, frm, to, default, resolution=1, tooltip=None):
        container = ttk.Frame(parent, style='M3.SurfaceContainer.TFrame')
        container.pack(fill='x', pady=(0, 16))
        header_frame = ttk.Frame(container, style='M3.SurfaceContainer.TFrame')
        header_frame.pack(fill='x', pady=(0, 8))
        ttk.Label(header_frame, text=label, style='M3.BodyMedium.TLabel').pack(side='left')
        formatted_default = f"{default:.2f}" if resolution < 1 else str(int(float(default)))
        value_label = ttk.Label(header_frame, text=formatted_default, style='M3.LabelLarge.TLabel')
        value_label.configure(foreground=self.theme.COLORS['primary'])
        value_label.pack(side='right')
        slider = ttk.Scale(container, from_=frm, to=to, orient='horizontal', style='M3.Horizontal.TScale')
        slider.set(default)
        slider.pack(fill='x')
        def update_label(v):
            val_str = f"{float(v):.2f}" if resolution < 1 else str(int(float(v)))
            value_label.config(text=val_str)
        slider.config(command=update_label)
        if tooltip:
            self._attach_tooltip(container, tooltip)
            self._attach_tooltip(slider, tooltip)
            self._attach_tooltip(value_label, tooltip)
        return slider

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-o>", lambda e: self.upload())
        self.root.bind_all("<Control-w>", lambda e: self.start_webcam())
        self.root.bind_all("<Control-s>", lambda e: self.save())
        self.root.bind_all("<space>", lambda e: self.process_btn.cget('state') == 'normal' and self.cartoonify_threaded())
        self.root.bind_all("<Escape>", lambda e: self.stop_webcam_flag.set())

    def upload(self, *args):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if path: self._load_image(path)

    def batch_select(self):
        paths = filedialog.askopenfilenames(title="Select images for batch processing")
        if not paths: return
        dest = filedialog.askdirectory(title="Choose output directory")
        if not dest: return
        threading.Thread(target=self._run_batch, args=(paths, dest), daemon=True).start()

    def _run_batch(self, paths, dest_folder):
        total = len(paths)
        if total == 0:
            return
        self.root.after(0, lambda: self._set_status(f"Processing {total} images...", progress=0))
        for idx, p in enumerate(paths, 1):
            try:
                base = os.path.basename(p)
                start_pct = int(((idx - 1) / total) * 100)
                self.root.after(0, lambda b=base, i=idx, pct=start_pct: self._set_status(f"Processing {i}/{total}: {b}", progress=pct))
                img_cv = cv2.imread(p)
                if img_cv is None: continue
                processed = self._process_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                outname, ext = os.path.splitext(base)
                outpath = os.path.join(dest_folder, f"{outname}_cartoon.png")
                cv2.imwrite(outpath, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                done_pct = int((idx / total) * 100)
                self.root.after(0, lambda b=base, i=idx, pct=done_pct: self._set_status(f"Completed {i}/{total}: {b}", progress=pct))
            except Exception as e:
                logging.error(f"Batch error on {p}: {e}")
        self.root.after(0, lambda: self._set_status("Batch processing complete.", progress=100))
        self.root.after(600, lambda: self._set_status(f"Batch processing of {total} images complete.", busy=False))
        self.root.after(0, lambda: messagebox.showinfo("Batch Complete", f"Processed {total} images."))

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files: self._load_image(files[0])

    def _load_image(self, path):
        try:
            img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            self.original_path, self.original_cv_image = path, img_rgb
            self.cartoon_cv_image, self.previous_cartoon_image = None, None
            self._update_before_preview(img_rgb)
            self.after_canvas.config(image='', text="Click 'Cartoonify' to process\n\n🎨")
            self._photo_after = None
            for btn in (self.process_btn, self.save_btn, self.compare_btn, self.undo_btn):
                btn.config(state='disabled')
            self.process_btn.config(state='normal')
            self._set_status(f"Loaded: {os.path.basename(path)}")
            self._schedule_live_preview(delay=500)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def _update_before_preview(self, img_rgb):
        self._update_preview(self.before_canvas, img_rgb)
        
    def _update_after_preview(self, img_rgb):
        self._update_preview(self.after_canvas, img_rgb)

    def _update_preview(self, canvas_widget, img_rgb):
        try:
            pil = pil_from_cv_rgb(img_rgb)
            pil.thumbnail(THUMBNAIL_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            canvas_widget.config(image=photo, text="")
            if canvas_widget == self.before_canvas: self._photo_before = photo
            else: self._photo_after = photo
        except Exception as e:
            canvas_widget.config(text="Preview failed", image="")
            logging.error(f"Preview update failed: {e}")

    def _set_status(self, text, busy=False, progress=None):
        """Update the status label and manage the shared progress bar.

        Args:
            text (str): Status message to display.
            busy (bool): Whether to show an indeterminate spinner.
            progress (Optional[float]): Determinate progress percentage (0-100).
        """
        self.status_label.config(text=text)

        if progress is not None:
            pct = max(0.0, min(100.0, float(progress)))
            if not self.progress_bar.winfo_ismapped():
                self.progress_bar.pack(fill='x', padx=24, pady=(0, 16))
                self._progress_current = 0.0
            current_mode = str(self.progress_bar.cget('mode'))
            if current_mode != 'determinate':
                self.progress_bar.stop()
                self.progress_bar.configure(mode='determinate', maximum=100.0, value=0.0)
                self._progress_current = 0.0
            
            # Start smooth animation to target value
            self._progress_target = pct
            if self._progress_animation_job is None:
                self._animate_progress()
            return

        if busy:
            if not self.progress_bar.winfo_ismapped():
                self.progress_bar.pack(fill='x', padx=24, pady=(0, 16))
            if str(self.progress_bar.cget('mode')) != 'indeterminate':
                self.progress_bar.configure(mode='indeterminate', maximum=100.0)
            self.progress_bar.start(12)
        else:
            if self.progress_bar.winfo_ismapped():
                self.progress_bar.stop()
                self.progress_bar.configure(mode='indeterminate', maximum=100.0)
                self.progress_bar['value'] = 0.0
                self.progress_bar.pack_forget()
                if self._progress_animation_job is not None:
                    self.root.after_cancel(self._progress_animation_job)
                    self._progress_animation_job = None

    def _animate_progress(self):
        """Smooth interpolation animation for progress bar."""
        try:
            if not self.progress_bar.winfo_exists() or not self.progress_bar.winfo_ismapped():
                self._progress_animation_job = None
                return
        except tk.TclError:
            self._progress_animation_job = None
            return
        
        # Smooth interpolation
        diff = self._progress_target - self._progress_current
        if abs(diff) < 0.5:
            self._progress_current = self._progress_target
            try:
                self.progress_bar['value'] = self._progress_current
            except tk.TclError:
                pass
            self._progress_animation_job = None
            return
        
        # Ease towards target
        self._progress_current += diff * 0.2
        try:
            self.progress_bar['value'] = self._progress_current
            self.root.update_idletasks()
        except tk.TclError:
            self._progress_animation_job = None
            return
        
        self._progress_animation_job = self.root.after(30, self._animate_progress)

    def _on_control_change(self, event=None):
        if event and event.widget == self.style_combobox:
            self._update_parameter_widgets(event)
        self._schedule_live_preview()

    def _schedule_live_preview(self, delay=250):
        if not self.live_preview_var.get() or self.original_cv_image is None:
            return
        if self.preview_debounce_job:
            self.root.after_cancel(self.preview_debounce_job)
            self.preview_debounce_job = None
        if self._preview_cancel_event:
            self._preview_cancel_event.set()
        token = threading.Event()
        self._preview_cancel_event = token
        self._start_preview_progress()
        self.preview_debounce_job = self.root.after(delay, lambda: self._run_live_preview(token))

    def _toggle_live_preview(self):
        if self.live_preview_var.get() and self.original_cv_image is not None:
            self._schedule_live_preview()
        elif not self.live_preview_var.get() and self.cartoon_cv_image is not None:
            if self._preview_cancel_event:
                self._preview_cancel_event.set()
            self._update_after_preview(self.cartoon_cv_image)
            self._stop_preview_progress(success=False, message="Live preview idle")

    def _run_live_preview(self, token):
        if token is None or token is not self._preview_cancel_event:
            return
        if token.is_set() or not self.live_preview_var.get() or self.original_cv_image is None:
            return

        def worker():
            if token.is_set():
                return
            try:
                h, w = self.original_cv_image.shape[:2]
                ch, cw = LIVE_PREVIEW_CROP
                x1, y1 = max(0, (w - cw) // 2), max(0, (h - ch) // 2)
                crop = self.original_cv_image[y1:y1+ch, x1:x1+cw].copy()
                face_mask = self._get_face_mask(crop) if self.face_focus_var.get() else None
                preview_img = self._process_image(crop, for_preview=True, face_mask_override=face_mask)
                if token.is_set():
                    return
                self.root.after(0, lambda: self._complete_live_preview(token, preview_img))
            except Exception as exc:
                logging.error(f"Live preview update failed: {exc}")
                self.root.after(0, lambda: self._handle_preview_error(token, "Preview update failed"))

        self._preview_executor.submit(worker)

    def _complete_live_preview(self, token, preview_img):
        if token is not self._preview_cancel_event or token.is_set():
            return
        self.preview_debounce_job = None
        self._update_after_preview(preview_img)
        self._stop_preview_progress(success=True)

    def _handle_preview_error(self, token, message):
        if token is not self._preview_cancel_event or token.is_set():
            return
        self.preview_debounce_job = None
        self._stop_preview_progress(success=False, message=message)
    
    def _update_parameter_widgets(self, event=None):
        selected_style = self.style_var.get()
        self.standard_params_frame.pack_forget()
        self.advanced_params_frame.pack_forget()
        if selected_style == "Cartoon V2 (Advanced)":
            self.advanced_params_frame.pack(fill='x')
            self.cartoon_tone_frame.pack_forget()
        elif selected_style == "Cartoon":
            self.standard_params_frame.pack(fill='x')
            if not self.cartoon_tone_frame.winfo_manager():
                self.cartoon_tone_frame.pack(fill='x', pady=(0, 16))
        elif selected_style in ["Pencil Sketch", "Comic Dots"]:
            self.standard_params_frame.pack(fill='x')
            self.cartoon_tone_frame.pack_forget()
        else:
            self.cartoon_tone_frame.pack_forget()

    def _get_face_mask(self, img_rgb):
        if self.face_cascade is None: return None
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0: return None
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in faces:
            ex, ey = w // 6, h // 6
            cv2.ellipse(mask, (x + w//2, y + h//2), (w//2 + ex, h//2 + ey), 0, 0, 360, 255, -1)
        return cv2.GaussianBlur(mask, (21, 21), 0)

    def _process_image(self, img_rgb, for_preview=False, face_mask_override=None):
        style = self.style_var.get()
        face_mask = face_mask_override if face_mask_override is not None and self.face_focus_var.get() else (self._get_face_mask(img_rgb) if self.face_focus_var.get() else None)
        try:
            if style == "Cartoon V2 (Advanced)":
                return ImageEffects.cartoon_effect_v2(img_rgb,
                    line_thickness=safe_int(self.line_thickness_scale.get()),
                    canny_low=safe_int(self.canny_low_scale.get()),
                    canny_high=safe_int(self.canny_high_scale.get()),
                    bilateral_d=ensure_odd(self.bilateral_d_scale.get()),
                    saturation_boost=safe_float(self.saturation_scale.get()),
                    value_boost=safe_float(self.value_boost_scale.get()),
                    tone_gamma=safe_float(self.tone_gamma_scale.get()),
                    local_contrast=safe_float(self.local_contrast_scale.get()),
                    adaptive_canny_mix=max(0.0, min(1.0, safe_float(self.adaptive_mix_scale.get()))),
                    sharpen_amount=safe_float(self.sharpen_scale.get()),
                    face_mask=face_mask)
            elif style == "Cartoon":
                return ImageEffects.cartoon_effect(
                    img_rgb,
                    edge_enhance=safe_int(self.edge_scale.get()),
                    color_levels=safe_int(self.color_scale.get()),
                    smooth_value=safe_int(self.smooth_scale.get()),
                    contrast=safe_float(self.contrast_scale.get()),
                    saturation_boost=safe_float(self.cartoon_saturation_scale.get()),
                    value_boost=safe_float(self.cartoon_value_scale.get()),
                    gamma=safe_float(self.cartoon_gamma_scale.get()),
                    local_contrast=safe_float(self.cartoon_local_contrast_scale.get()),
                    face_mask=face_mask)
            elif style == "Pencil Sketch":
                return ImageEffects.pencil_sketch(img_rgb, ksize=ensure_odd(self.smooth_scale.get()), contrast=safe_float(self.contrast_scale.get()))
            elif style == "Watercolor":
                return ImageEffects.watercolor(img_rgb, sigma_s=40 if for_preview else 60, sigma_r=0.7 if for_preview else 0.6)
            elif style == "Comic Dots":
                return ImageEffects.comic_dots(img_rgb, dot_scale=max(4, safe_int(self.color_scale.get()) // 2))
            return img_rgb
        except Exception as e:
            logging.error(f"Processing error: {e}")
            return img_rgb

    def cartoonify_threaded(self):
        if self.original_cv_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        self.process_btn.config(state='disabled', text='Processing...')
        self.save_btn.config(state='disabled')
        self._set_status("Processing your image...", progress=5)
        
        threading.Thread(target=self._cartoonify_worker, daemon=True).start()

    def _cartoonify_worker(self):
        try:
            if self.cartoon_cv_image is not None:
                self.previous_cartoon_image = self.cartoon_cv_image.copy()
            self.root.after(0, lambda: self._set_status("Preparing filters...", progress=20))
            processed = self._process_image(self.original_cv_image)
            self.cartoon_cv_image = processed
            self.root.after(0, lambda: self._set_status("Finishing touches...", progress=80))
            self.root.after(0, lambda: self._on_processing_done(processed))
        except Exception as e:
            self.root.after(0, lambda: self._on_processing_failed(str(e)))

    def _on_processing_done(self, processed):
        self.process_btn.config(state='normal', text='Cartoonify')
        self.save_btn.config(state='normal')
        self.compare_btn.config(state='normal')
        if self.previous_cartoon_image is not None: self.undo_btn.config(state='normal')
        self._update_after_preview(processed)
        self._set_status("Processing complete. Ready to save.", progress=100)
        self.root.after(400, lambda: self._set_status("Processing complete. Ready to save.", busy=False))

    def _on_processing_failed(self, msg):
        self.process_btn.config(state='normal', text='Cartoonify')
        self._set_status("Processing failed. Please try again.", busy=False)
        messagebox.showerror("Processing Error", f"Failed to process image: {msg}")

    def save(self):
        if self.cartoon_cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="cartoon.png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path: return
        try:
            cv2.imwrite(path, cv2.cvtColor(self.cartoon_cv_image, cv2.COLOR_RGB2BGR))
            self._set_status(f"Image saved to {os.path.basename(path)}", busy=False)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")

    def export_comparison(self):
        if self.original_cv_image is None or self.cartoon_cv_image is None: return
        orig, proc = pil_from_cv_rgb(self.original_cv_image), pil_from_cv_rgb(self.cartoon_cv_image)
        h = max(orig.height, proc.height)
        orig_resized = orig.resize((int(orig.width * h / orig.height), h), Image.Resampling.LANCZOS)
        proc_resized = proc.resize((int(proc.width * h / proc.height), h), Image.Resampling.LANCZOS)
        combined = Image.new('RGB', (orig_resized.width + proc_resized.width, h))
        combined.paste(orig_resized, (0, 0))
        combined.paste(proc_resized, (orig_resized.width, 0))
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="comparison.png")
        if path: combined.save(path)

    def undo(self):
        if self.previous_cartoon_image is None: return
        self.cartoon_cv_image, self.previous_cartoon_image = self.previous_cartoon_image, None
        self.undo_btn.config(state='disabled')
        self._update_after_preview(self.cartoon_cv_image)
        self._set_status("Undo successful.")

    def reset_to_defaults(self):
        mappings = [
            (self.edge_scale, 'edge_enhance'),
            (self.color_scale, 'color_levels'),
            (self.smooth_scale, 'smooth'),
            (self.contrast_scale, 'contrast'),
            (self.cartoon_saturation_scale, 'cartoon_saturation'),
            (self.cartoon_value_scale, 'cartoon_value'),
            (self.cartoon_gamma_scale, 'cartoon_gamma'),
            (self.cartoon_local_contrast_scale, 'cartoon_local_contrast'),
            (self.line_thickness_scale, 'line_thickness'),
            (self.canny_low_scale, 'canny_low'),
            (self.canny_high_scale, 'canny_high'),
            (self.bilateral_d_scale, 'bilateral_d'),
            (self.saturation_scale, 'advanced_saturation'),
            (self.value_boost_scale, 'advanced_value'),
            (self.tone_gamma_scale, 'advanced_gamma'),
            (self.local_contrast_scale, 'advanced_local_contrast'),
            (self.adaptive_mix_scale, 'adaptive_mix'),
            (self.sharpen_scale, 'sharpen'),
        ]
        for widget, key in mappings:
            widget.set(self.defaults[key])
        self.style_var.set(self.defaults['style'])
        self.face_focus_var.set(self.defaults['face_focus'])
        self._update_parameter_widgets()
        self._set_status("Parameters reset to defaults.")

    def start_webcam(self):
        self.stop_webcam_flag.clear()
        self.current_webcam_frame = None  # Store the latest processed frame
        
        cam_win = tk.Toplevel(self.root)
        cam_win.title("Webcam Cartoonify")
        cam_win.geometry("1000x750")
        cam_win.configure(bg=self.theme.COLORS['surface'])
        
        # Create frame for video display
        video_frame = ttk.Frame(cam_win, style='M3.Surface.TFrame')
        video_frame.pack(fill='both', expand=True, padx=16, pady=(16, 8))
        
        label = tk.Label(video_frame, bg=self.theme.COLORS['surface'])
        label.pack(fill='both', expand=True)
        
        # Create button bar at the bottom
        button_bar = ttk.Frame(cam_win, style='M3.SurfaceContainer.TFrame')
        button_bar.pack(fill='x', padx=16, pady=(8, 16))
        
        # Capture button
        capture_btn = ttk.Button(
            button_bar, 
            text="📸 Capture & Save", 
            command=lambda: self._capture_webcam_frame(cam_win),
            style='M3.Filled.TButton'
        )
        capture_btn.pack(side='left', padx=(0, 8), pady=8)
        
        # Close button
        close_btn = ttk.Button(
            button_bar,
            text="Close",
            command=lambda: [self.stop_webcam_flag.set(), cam_win.destroy()],
            style='M3.Outlined.TButton'
        )
        close_btn.pack(side='right', padx=(8, 0), pady=8)
        
        # Status label
        self.webcam_status = ttk.Label(
            button_bar,
            text="Live preview active",
            style='M3.BodyMedium.TLabel'
        )
        self.webcam_status.pack(side='left', padx=16, pady=8)
        
        threading.Thread(target=self._webcam_loop, args=(cam_win, label), daemon=True).start()
        cam_win.protocol("WM_DELETE_WINDOW", lambda: [self.stop_webcam_flag.set(), cam_win.destroy()])

    def _webcam_loop(self, win, label_widget):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Webcam Error", "Cannot access webcam."))
            self.root.after(0, win.destroy)
            return
        
        photo_ref = None # To prevent garbage collection
        while not self.stop_webcam_flag.is_set():
            try:
                ret, frame = cap.read()
                if not ret: 
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self._process_image(frame_rgb, for_preview=True)
                
                # Store the latest processed frame for capture
                self.current_webcam_frame = processed
                
                pil = pil_from_cv_rgb(processed)
                
                # Check if widget still exists before accessing it
                try:
                    w, h = label_widget.winfo_width(), label_widget.winfo_height()
                    if w > 1 and h > 1:
                        pil.thumbnail((w, h), Image.Resampling.LANCZOS)
                    photo_ref = ImageTk.PhotoImage(pil)
                    self.root.after(0, label_widget.config, {'image': photo_ref})
                except tk.TclError:
                    # Widget was destroyed, stop the loop
                    break
                    
                time.sleep(0.03) # ~30 FPS
                
            except Exception as e:
                logging.error(f"Error in webcam loop: {e}")
                break
                
        cap.release()
        # Only try to destroy window if it still exists
        try:
            if win.winfo_exists():
                self.root.after(0, win.destroy)
        except tk.TclError:
            pass
    
    def _capture_webcam_frame(self, cam_window):
        """Capture and save the current webcam frame."""
        if self.current_webcam_frame is None:
            messagebox.showwarning("No Frame", "No frame available to capture.")
            return
        
        # Ask user where to save
        file_path = filedialog.asksaveasfilename(
            parent=cam_window,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*")
            ],
            title="Save Captured Frame"
        )
        
        if not file_path:
            return
        
        try:
            # Convert RGB to PIL and save
            pil_image = pil_from_cv_rgb(self.current_webcam_frame)
            pil_image.save(file_path, quality=95)
            
            # Update status
            if hasattr(self, 'webcam_status'):
                self.webcam_status.config(text=f"✓ Saved: {os.path.basename(file_path)}")
                # Reset status after 3 seconds
                self.root.after(3000, lambda: self.webcam_status.config(text="Live preview active"))
            
            messagebox.showinfo("Success", f"Frame saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save frame:\n{str(e)}")

# --- Application Entry Point ---
def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = CartoonifyStudio(root)
    root.mainloop()

if __name__ == "__main__":
    main()