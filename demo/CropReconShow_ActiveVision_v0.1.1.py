"""
v0.1.1 - 2024-06-17
MNIST Patch-Capture Program using Tkinter.

This app displays an MNIST image, a square crop, and resized/reconstructed views.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict

import numpy as np
from PIL import Image, ImageTk
from torchvision import datasets

RESAMPLING = getattr(Image, "Resampling", Image)


def create_image_state(image: np.ndarray, is_grayscale: bool) -> Dict[str, object]:
    """
    Wrap image data with display metadata used by the UI.
    """
    return {"data": image, "is_grayscale": is_grayscale}


def load_mnist_sample(index: int = 0) -> Dict[str, object]:
    """
    Load a single MNIST image.
    """
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    image, _ = dataset[index]
    return create_image_state(np.array(image).astype(np.float32) / 255.0, True)


def load_random_mnist_sample() -> Dict[str, object]:
    """
    Load a random MNIST image from the training split.
    """
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    random_index = np.random.randint(0, len(dataset))
    image, _ = dataset[random_index]
    return create_image_state(np.array(image).astype(np.float32) / 255.0, True)


def extract_square_patch(
    image: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    padding_mode: str = "zero",
) -> np.ndarray:
    """
    Extract a square patch from the image centered at (cx, cy).
    """
    img_h, img_w = image.shape[:2]

    x_start = int(np.floor(cx - size / 2))
    y_start = int(np.floor(cy - size / 2))
    x_end = x_start + size
    y_end = y_start + size

    patch_shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    patch = np.zeros(patch_shape, dtype=image.dtype)

    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(img_w, x_end)
    src_y_end = min(img_h, y_end)

    if src_x_start < src_x_end and src_y_start < src_y_end:
        dest_x_start = src_x_start - x_start
        dest_y_start = src_y_start - y_start
        dest_x_end = dest_x_start + (src_x_end - src_x_start)
        dest_y_end = dest_y_start + (src_y_end - src_y_start)

        patch[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = image[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return patch


def resize_patch(patch: np.ndarray, out_size: int) -> np.ndarray:
    """
    Resize a square patch to (out_size, out_size) using bilinear interpolation.
    """
    if out_size == 1:
        if patch.ndim == 2:
            return np.array([[np.mean(patch)]], dtype=patch.dtype)
        return np.mean(patch, axis=(0, 1), keepdims=True).astype(patch.dtype)

    src_h, src_w = patch.shape[:2]
    resized_shape = (
        (out_size, out_size) if patch.ndim == 2 else (out_size, out_size, patch.shape[2])
    )
    resized = np.zeros(resized_shape, dtype=patch.dtype)
    scale = (src_h - 1) / (out_size - 1) if out_size > 1 else 0

    for r in range(out_size):
        for c in range(out_size):
            src_r = r * scale
            src_c = c * scale

            r0 = int(np.floor(src_r))
            r1 = min(r0 + 1, src_h - 1)
            c0 = int(np.floor(src_c))
            c1 = min(c0 + 1, src_w - 1)

            dr = src_r - r0
            dc = src_c - c0

            val = (
                (1 - dr) * (1 - dc) * patch[r0, c0]
                + dr * (1 - dc) * patch[r1, c0]
                + (1 - dr) * dc * patch[r0, c1]
                + dr * dc * patch[r1, c1]
            )

            resized[r, c] = val

    return resized


def build_reconstructed_view(
    image: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    resized_patch: np.ndarray,
) -> np.ndarray:
    """
    Create a black image the same size as the original and place an enlarged
    version of the resized patch into the selected crop area.
    """
    output = np.zeros_like(image)
    enlarged = array_to_pil_image(resized_patch).resize((size, size), RESAMPLING.NEAREST)
    enlarged_array = np.array(enlarged).astype(np.float32) / 255.0

    if image.ndim == 2 and enlarged_array.ndim == 3:
        enlarged_array = enlarged_array[..., 0]
    if image.ndim == 3 and enlarged_array.ndim == 2:
        enlarged_array = np.repeat(enlarged_array[..., None], image.shape[2], axis=2)

    img_h, img_w = image.shape[:2]
    x_start = int(np.floor(cx - size / 2))
    y_start = int(np.floor(cy - size / 2))
    x_end = x_start + size
    y_end = y_start + size

    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(img_w, x_end)
    src_y_end = min(img_h, y_end)

    if src_x_start < src_x_end and src_y_start < src_y_end:
        patch_x_start = src_x_start - x_start
        patch_y_start = src_y_start - y_start
        patch_x_end = patch_x_start + (src_x_end - src_x_start)
        patch_y_end = patch_y_start + (src_y_end - src_y_start)
        output[src_y_start:src_y_end, src_x_start:src_x_end] = enlarged_array[
            patch_y_start:patch_y_end, patch_x_start:patch_x_end
        ]

    return output


def get_crop_bounds(image: np.ndarray, cx: float, cy: float, size: int) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Return the valid destination bounds in the original image and matching source
    bounds inside a square crop-sized patch.
    """
    img_h, img_w = image.shape[:2]
    x_start = int(np.floor(cx - size / 2))
    y_start = int(np.floor(cy - size / 2))
    x_end = x_start + size
    y_end = y_start + size

    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(img_w, x_end)
    src_y_end = min(img_h, y_end)

    patch_x_start = src_x_start - x_start
    patch_y_start = src_y_start - y_start
    patch_x_end = patch_x_start + max(0, src_x_end - src_x_start)
    patch_y_end = patch_y_start + max(0, src_y_end - src_y_start)

    return (
        src_x_start,
        src_y_start,
        src_x_end,
        src_y_end,
        patch_x_start,
        patch_y_start,
        patch_x_end,
        patch_y_end,
    )


def array_to_pil_image(image: np.ndarray) -> Image.Image:
    """
    Convert a normalized NumPy array to a PIL image.
    """
    clipped = np.clip(image, 0.0, 1.0)
    pixels = (clipped * 255).astype(np.uint8)
    if image.ndim == 2:
        return Image.fromarray(pixels, mode="L")
    return Image.fromarray(pixels, mode="RGB")


def black_image_like(image: np.ndarray) -> np.ndarray:
    """
    Return a black image with the same shape as the input.
    """
    return np.zeros_like(image)


class ImageCanvas(ttk.Frame):
    """
    Resizable image display panel with optional crop overlay.
    """

    def __init__(self, master: tk.Widget, title: str) -> None:
        super().__init__(master, padding=6)
        self.title_var = tk.StringVar(value=title)
        self.canvas = tk.Canvas(self, bg="#1f1f1f", highlightthickness=0)
        self.label = ttk.Label(self, textvariable=self.title_var, anchor="center")
        self.canvas.pack(fill="both", expand=True)
        self.label.pack(fill="x", pady=(6, 0))

        self._image = None
        self._rect = None
        self._overlay_text = None
        self._photo = None
        self._canvas_image_id = None
        self._display_scale = 1.0
        self._display_offset_x = 0
        self._display_offset_y = 0
        self._display_width = 1
        self._display_height = 1
        self.canvas.bind("<Configure>", self._on_resize)

    def set_content(
        self,
        image: np.ndarray,
        title: str,
        rect: tuple[float, float, float] | None = None,
        overlay_text: str | None = None,
    ) -> None:
        self._image = image
        self._rect = rect
        self._overlay_text = overlay_text
        self.title_var.set(title)
        self._redraw()

    def _on_resize(self, _event: tk.Event) -> None:
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._image is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        pil_image = array_to_pil_image(self._image)
        img_w, img_h = pil_image.size

        scale = min(canvas_w / img_w, canvas_h / img_h)
        display_w = max(1, int(img_w * scale))
        display_h = max(1, int(img_h * scale))
        resized = pil_image.resize((display_w, display_h), RESAMPLING.NEAREST)

        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        self._display_scale = scale
        self._display_offset_x = offset_x
        self._display_offset_y = offset_y
        self._display_width = display_w
        self._display_height = display_h

        self._photo = ImageTk.PhotoImage(resized)
        self._canvas_image_id = self.canvas.create_image(
            offset_x, offset_y, anchor="nw", image=self._photo
        )

        if self._rect is not None:
            cx, cy, size = self._rect
            left = offset_x + (cx - size / 2) * scale
            top = offset_y + (cy - size / 2) * scale
            right = offset_x + (cx + size / 2) * scale
            bottom = offset_y + (cy + size / 2) * scale
            self.canvas.create_rectangle(left, top, right, bottom, outline="#ff4d4d", width=2)
            if self._overlay_text:
                text_x = (left + right) / 2
                text_y = max(offset_y + 12, top - 12)
                self.canvas.create_text(
                    text_x,
                    text_y,
                    text=self._overlay_text,
                    fill="#32cd32",
                    font=("Helvetica", 12, "bold"),
                )

    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> tuple[float, float] | None:
        """
        Convert canvas coordinates to image coordinates if the cursor is over the displayed image.
        """
        if self._image is None:
            return None

        local_x = canvas_x - self._display_offset_x
        local_y = canvas_y - self._display_offset_y
        if local_x < 0 or local_y < 0 or local_x > self._display_width or local_y > self._display_height:
            return None

        img_h, img_w = self._image.shape[:2]
        image_x = local_x / self._display_scale
        image_y = local_y / self._display_scale
        image_x = min(max(image_x, 0.0), float(img_w))
        image_y = min(max(image_y, 0.0), float(img_h))
        return image_x, image_y


class MNISTPatchApp:
    """
    Tkinter application for interactive MNIST and image patch capture.
    """

    def __init__(self, root: tk.Tk, image_state: Dict[str, object]) -> None:
        self.root = root
        self.root.title("MNIST Patch-Capture Program")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        self.root.minsize(900, 560)

        self.current_image = image_state.copy()
        self.is_updating = False
        self.active_mouse_panel: int | None = None
        self.memory_image: np.ndarray | None = None
        self.remember_count = 0
        self.accumulated_pixel_count = 0

        self._build_layout()
        self._bind_keyboard_controls()
        self._set_image_state(self.current_image, reset_controls=True)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        display_frame = ttk.Frame(main)
        display_frame.grid(row=0, column=0, sticky="nsew")
        for col in range(5):
            display_frame.columnconfigure(col, weight=1, uniform="display")
        display_frame.rowconfigure(0, weight=1)

        self.original_panel = ImageCanvas(display_frame, "Original")
        self.crop_panel = ImageCanvas(display_frame, "Cropped")
        self.resized_panel = ImageCanvas(display_frame, "Resized")
        self.reconstructed_panel = ImageCanvas(display_frame, "Reconstructed")
        self.memory_panel = ImageCanvas(display_frame, "Remembered")
        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.crop_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        self.resized_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.reconstructed_panel.grid(row=0, column=3, sticky="nsew", padx=6)
        self.memory_panel.grid(row=0, column=4, sticky="nsew", padx=(6, 0))
        self.original_panel.canvas.bind("<Enter>", lambda event: self._on_panel_enter(event, 1))
        self.original_panel.canvas.bind("<Leave>", self._on_panel_leave)
        self.original_panel.canvas.bind("<Motion>", self._on_pointer_panel_1_or_4)
        self.reconstructed_panel.canvas.bind("<Enter>", lambda event: self._on_panel_enter(event, 4))
        self.reconstructed_panel.canvas.bind("<Leave>", self._on_panel_leave)
        self.reconstructed_panel.canvas.bind("<Motion>", self._on_pointer_panel_1_or_4)
        for canvas in (self.original_panel.canvas, self.reconstructed_panel.canvas):
            canvas.bind("<MouseWheel>", self._on_mouse_wheel)
            canvas.bind("<Button-4>", self._on_mouse_wheel)
            canvas.bind("<Button-5>", self._on_mouse_wheel)
            canvas.bind("<Button-1>", self._on_left_click_panel_1_or_4)
            canvas.bind("<Button-3>", self._on_right_click_panel_1_or_4)

        controls = ttk.Frame(main, padding=(0, 12, 0, 0))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        button_row = ttk.Frame(controls)
        button_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Button(button_row, text="Change Image", command=self._change_image).pack(side="left")
        ttk.Button(button_row, text="Remember", command=self._remember_current_crop).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Reset", command=self._reset_memory).pack(side="left", padx=(8, 0))

        checkbox_row = ttk.Frame(controls)
        checkbox_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.panel_mask_vars = [tk.BooleanVar(value=False) for _ in range(5)]
        ttk.Checkbutton(
            checkbox_row,
            text="Blackout Panel 1",
            variable=self.panel_mask_vars[0],
            command=self._render_all,
        ).pack(side="left")
        ttk.Checkbutton(
            checkbox_row,
            text="Blackout Panel 2",
            variable=self.panel_mask_vars[1],
            command=self._render_all,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            checkbox_row,
            text="Blackout Panel 3",
            variable=self.panel_mask_vars[2],
            command=self._render_all,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            checkbox_row,
            text="Blackout Panel 4",
            variable=self.panel_mask_vars[3],
            command=self._render_all,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            checkbox_row,
            text="Blackout Panel 5",
            variable=self.panel_mask_vars[4],
            command=self._render_all,
        ).pack(side="left", padx=(12, 0))

        self.cx_var = tk.DoubleVar()
        self.cy_var = tk.DoubleVar()
        self.size_var = tk.IntVar()
        self.output_var = tk.IntVar()
        self.size_note_var = tk.StringVar(value="")
        self.remember_note_var = tk.StringVar(value="Remember Count: 0")
        self.pixel_note_var = tk.StringVar(value="Accumulated Pixels: 0")

        ttk.Label(controls, textvariable=self.size_note_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, textvariable=self.remember_note_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, textvariable=self.pixel_note_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.cx_scale = self._add_scale_row(controls, 5, "Center X", self.cx_var, self._on_control_change)
        self.cy_scale = self._add_scale_row(controls, 6, "Center Y", self.cy_var, self._on_control_change)
        self.size_scale = self._add_scale_row(controls, 7, "Crop Size (S)", self.size_var, self._on_control_change)
        self.output_scale = self._add_scale_row(controls, 8, "Output (N)", self.output_var, self._on_control_change)

    def _bind_keyboard_controls(self) -> None:
        self.root.bind("<Up>", self._on_arrow_key)
        self.root.bind("<Down>", self._on_arrow_key)
        self.root.bind("<Left>", self._on_arrow_key)
        self.root.bind("<Right>", self._on_arrow_key)

    def _add_scale_row(
        self,
        parent: ttk.Frame,
        row: int,
        label_text: str,
        variable: tk.Variable,
        command,
    ) -> tk.Scale:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        scale = tk.Scale(
            parent,
            from_=0,
            to=1,
            orient="horizontal",
            resolution=1,
            showvalue=True,
            variable=variable,
            command=command,
        )
        scale.grid(row=row, column=1, sticky="ew", pady=4)
        return scale

    def _get_image_shape(self) -> tuple[int, int]:
        image = self.current_image["data"]
        return image.shape[0], image.shape[1]

    def _get_max_square_size(self) -> int:
        img_h, img_w = self._get_image_shape()
        return max(1, max(img_h, img_w))

    def _configure_scale(self, scale: tk.Scale, minimum: int, maximum: int, resolution: int = 1) -> None:
        scale.configure(from_=minimum, to=maximum, resolution=resolution)

    def _set_image_state(self, image_state: Dict[str, object], reset_controls: bool) -> None:
        self.current_image = image_state.copy()
        img_h, img_w = self._get_image_shape()
        max_square = self._get_max_square_size()
        self.memory_image = black_image_like(self.current_image["data"])
        self.remember_count = 0
        self.accumulated_pixel_count = 0
        self.remember_note_var.set("Remember Count: 0")
        self.pixel_note_var.set("Accumulated Pixels: 0")

        self.is_updating = True
        self._configure_scale(self.cx_scale, 0, img_w, 1)
        self._configure_scale(self.cy_scale, 0, img_h, 1)
        self._configure_scale(self.size_scale, 1, max_square, 1)
        self._configure_scale(self.output_scale, 1, max_square, 1)

        if reset_controls:
            self.cx_var.set(img_w / 2)
            self.cy_var.set(img_h / 2)
            self.size_var.set(min(max_square, max(1, max_square // 2)))
            self.output_var.set(min(max_square, 4))
        else:
            self.cx_var.set(min(max(self.cx_var.get(), 0), img_w))
            self.cy_var.set(min(max(self.cy_var.get(), 0), img_h))
            self.size_var.set(min(max(self.size_var.get(), 1), max_square))
            self.output_var.set(min(max(self.output_var.get(), 1), max_square))
        self.is_updating = False

        self._render_all()

    def _change_image(self) -> None:
        self._set_image_state(load_random_mnist_sample(), reset_controls=True)

    def _remember_current_crop(self) -> None:
        image = self.current_image["data"]
        cx = float(self.cx_var.get())
        cy = float(self.cy_var.get())
        size = int(self.size_var.get())
        output_size = int(self.output_var.get())

        if self.memory_image is None or self.memory_image.shape != image.shape:
            self.memory_image = black_image_like(image)

        patch = extract_square_patch(image, cx, cy, size)
        resized = resize_patch(patch, output_size)
        reconstructed = build_reconstructed_view(image, cx, cy, size, resized)
        (
            src_x_start,
            src_y_start,
            src_x_end,
            src_y_end,
            _patch_x_start,
            _patch_y_start,
            _patch_x_end,
            _patch_y_end,
        ) = get_crop_bounds(image, cx, cy, size)
        self.memory_image[src_y_start:src_y_end, src_x_start:src_x_end] = reconstructed[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]
        self.remember_count += 1
        self.accumulated_pixel_count += output_size * output_size
        self.remember_note_var.set(f"Remember Count: {self.remember_count}")
        self.pixel_note_var.set(f"Accumulated Pixels: {self.accumulated_pixel_count}")
        self._render_all()

    def _reset_memory(self) -> None:
        self.memory_image = black_image_like(self.current_image["data"])
        self.remember_count = 0
        self.accumulated_pixel_count = 0
        self.remember_note_var.set("Remember Count: 0")
        self.pixel_note_var.set("Accumulated Pixels: 0")
        self._render_all()

    def _on_control_change(self, _value: str) -> None:
        if not self.is_updating:
            self._render_all()

    def _on_panel_enter(self, event: tk.Event, panel_index: int) -> None:
        self.active_mouse_panel = panel_index
        self._on_pointer_panel_1_or_4(event)

    def _on_panel_leave(self, _event: tk.Event) -> None:
        self.active_mouse_panel = None

    def _on_pointer_panel_1_or_4(self, event: tk.Event) -> None:
        panel = self.original_panel if event.widget is self.original_panel.canvas else self.reconstructed_panel
        coords = panel.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        image_x, image_y = coords
        self.is_updating = True
        self.cx_var.set(image_x)
        self.cy_var.set(image_y)
        self.is_updating = False
        self._render_all()

    def _on_left_click_panel_1_or_4(self, event: tk.Event) -> None:
        self._on_pointer_panel_1_or_4(event)
        if self.active_mouse_panel in (1, 4):
            self._remember_current_crop()

    def _on_right_click_panel_1_or_4(self, event: tk.Event) -> None:
        self._on_pointer_panel_1_or_4(event)
        if self.active_mouse_panel in (1, 4):
            self._reset_memory()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self.active_mouse_panel not in (1, 4):
            return

        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            delta = 1
        elif getattr(event, "num", None) == 5:
            delta = -1

        if delta == 0:
            return

        if self.active_mouse_panel == 1:
            current = int(self.size_var.get())
            minimum = int(float(self.size_scale.cget("from")))
            maximum = int(float(self.size_scale.cget("to")))
            new_value = current + delta
            new_value = min(max(new_value, minimum), maximum)
            if new_value != current:
                self.is_updating = True
                self.size_var.set(new_value)
                self.is_updating = False
                self._render_all()
        elif self.active_mouse_panel == 4:
            current = int(self.output_var.get())
            minimum = int(float(self.output_scale.cget("from")))
            maximum = int(float(self.output_scale.cget("to")))
            new_value = current + delta
            new_value = min(max(new_value, minimum), maximum)
            if new_value != current:
                self.is_updating = True
                self.output_var.set(new_value)
                self.is_updating = False
                self._render_all()

    def _on_arrow_key(self, event: tk.Event) -> None:
        if self.active_mouse_panel not in (1, 4):
            return

        if event.keysym == "Up":
            self._adjust_scale_value(self.size_var, self.size_scale, 1)
        elif event.keysym == "Down":
            self._adjust_scale_value(self.size_var, self.size_scale, -1)
        elif event.keysym == "Left":
            self._adjust_scale_value(self.output_var, self.output_scale, -1)
        elif event.keysym == "Right":
            self._adjust_scale_value(self.output_var, self.output_scale, 1)

    def _adjust_scale_value(self, variable: tk.Variable, scale: tk.Scale, step: int) -> None:
        current = int(variable.get())
        minimum = int(float(scale.cget("from")))
        maximum = int(float(scale.cget("to")))
        new_value = min(max(current + step, minimum), maximum)
        if new_value == current:
            return

        self.is_updating = True
        variable.set(new_value)
        self.is_updating = False
        self._render_all()

    def _render_all(self) -> None:
        image = self.current_image["data"]
        cx = float(self.cx_var.get())
        cy = float(self.cy_var.get())
        size = int(self.size_var.get())
        output_size = int(self.output_var.get())
        # size_note = f"{size} × {size} ⟶ {output_size} × {output_size}"
        size_note = f"[{size} x {size}] ==> [{output_size} x {output_size}]"
        self.size_note_var.set(f"Crop Size to N Size: {size_note}")

        patch = extract_square_patch(image, cx, cy, size)
        resized = resize_patch(patch, output_size)
        reconstructed = build_reconstructed_view(image, cx, cy, size, resized)
        memory_view = self.memory_image if self.memory_image is not None else black_image_like(image)

        panel_1_image = black_image_like(image) if self.panel_mask_vars[0].get() else image
        panel_2_image = black_image_like(patch) if self.panel_mask_vars[1].get() else patch
        panel_3_image = black_image_like(resized) if self.panel_mask_vars[2].get() else resized
        panel_4_image = black_image_like(reconstructed) if self.panel_mask_vars[3].get() else reconstructed
        panel_5_image = black_image_like(memory_view) if self.panel_mask_vars[4].get() else memory_view

        self.original_panel.set_content(
            panel_1_image,
            f"Original ({image.shape[1]}x{image.shape[0]})",
            rect=(cx, cy, size),
            overlay_text=size_note,
        )
        self.crop_panel.set_content(panel_2_image, f"Cropped {size}x{size}")
        self.resized_panel.set_content(panel_3_image, f"Resized {output_size}x{output_size}")
        self.reconstructed_panel.set_content(
            panel_4_image,
            f"Reconstructed Crop ({image.shape[1]}x{image.shape[0]})",
            rect=(cx, cy, size),
            overlay_text=size_note,
        )
        self.memory_panel.set_content(
            panel_5_image,
            f"Remembered ({image.shape[1]}x{image.shape[0]})",
        )


def main() -> None:
    root = tk.Tk()
    app = MNISTPatchApp(root, load_mnist_sample(0))
    root.mainloop()


if __name__ == "__main__":
    main()
