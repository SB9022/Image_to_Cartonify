# image_effects.py
"""
Image Processing Algorithms for Cartoonify Studio.
This module contains the core functions for applying artistic effects to images.
Author: (Your Name)
Date: 2025-09-24
"""

import logging
import math

import cv2
import numpy as np

def ensure_odd(v):
    """Ensures a value is an odd integer, which is required for some cv2 functions."""
    v = int(v)
    return v if v % 2 == 1 else (v + 1)

_STRUCTURING_ELEMENT_CACHE: dict[int, np.ndarray] = {}
_HALFTONE_DISTANCE_CACHE: dict[int, np.ndarray] = {}


def _validate_rgb_image(img_rgb):
    """Ensure the provided image is an 8-bit, 3-channel numpy array."""
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("Expected image input as numpy.ndarray")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img_rgb


def _resolve_bilateral_params(shape, base_kernel):
    """Scale bilateral filter parameters based on image resolution."""
    longest = max(shape[0], shape[1])
    scale = max(0.6, min(1.8, longest / 720.0))
    diameter = ensure_odd(max(3, int(round(base_kernel * scale))))
    sigma = 40.0 * scale
    return diameter, sigma, sigma


def _get_structuring_element(size):
    size = int(max(1, round(size)))
    if size <= 1:
        return None
    key = size
    if key not in _STRUCTURING_ELEMENT_CACHE:
        kernel_size = (size, size)
        _STRUCTURING_ELEMENT_CACHE[key] = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return _STRUCTURING_ELEMENT_CACHE[key]


def _adaptive_canny(gray_blur, base_low, base_high, mix=0.45):
    median = float(np.median(gray_blur))
    sigma = 0.33
    auto_low = int(max(0, (1.0 - sigma) * median))
    auto_high = int(min(255, (1.0 + sigma) * median))
    final_low = int(base_low * (1.0 - mix) + auto_low * mix)
    final_high = int(base_high * (1.0 - mix) + auto_high * mix)
    if final_low >= final_high:
        final_low = max(0, final_high - 10)
    return max(0, final_low), min(255, final_high)


def _apply_gamma(image, gamma):
    if gamma == 1.0:
        return image
    gamma = max(0.05, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def _apply_local_contrast(image, strength):
    if strength <= 0.0:
        return image
    clip = max(1.0, float(strength) * 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _prepare_halftone_distances(step):
    key = int(step)
    if key not in _HALFTONE_DISTANCE_CACHE:
        center = (step - 1) / 2.0
        yy = np.arange(step, dtype=np.float32)[:, None] - center
        xx = np.arange(step, dtype=np.float32)[None, :] - center
        _HALFTONE_DISTANCE_CACHE[key] = np.sqrt(yy ** 2 + xx ** 2)
    return _HALFTONE_DISTANCE_CACHE[key]


def _watercolor_fallback(img_rgb):
    temp = img_rgb.copy()
    for _ in range(2):
        temp = cv2.bilateralFilter(temp, 9, 75, 75)
    temp = cv2.medianBlur(temp, 3)
    return temp


class ImageEffects:
    """A collection of static methods for applying various image effects."""

    @staticmethod
    def cartoon_effect(
        img_rgb,
        edge_enhance=2,
        color_levels=6,
        smooth_value=7,
        contrast=1.1,
        saturation_boost=1.05,
        value_boost=1.02,
        gamma=1.0,
        local_contrast=0.0,
        face_mask=None,
    ):
        """Applies a standard cartoon effect to an RGB image."""
        img_rgb = _validate_rgb_image(img_rgb)
        img = img_rgb.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        d_gray, sigma_gray, sigma_space_gray = _resolve_bilateral_params(gray.shape, 5)
        gray_smooth = cv2.bilateralFilter(gray, d_gray, sigma_gray, sigma_space_gray)
        edges = cv2.adaptiveThreshold(gray_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 7, 7)
        if edge_enhance > 1:
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=int(edge_enhance))

        smooth_val = ensure_odd(smooth_value)
        diameter, sigma_color, sigma_space = _resolve_bilateral_params(img.shape, smooth_val)
        smooth = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        k = max(2, int(color_levels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        attempts = 10
        try:
            _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            quantized = centers[labels.flatten()].reshape(smooth.shape)
        except Exception:
            quantized = (np.floor(smooth / (256 // k)) * (256 // k)).astype(np.uint8)

        cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
        cartoon = cartoon.astype(np.float32)
        cartoon = cartoon * float(contrast)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)

        if saturation_boost != 1.0 or value_boost != 1.0:
            cartoon_hsv = cv2.cvtColor(cartoon, cv2.COLOR_RGB2HSV)
            if saturation_boost != 1.0:
                cartoon_hsv[:, :, 1] = np.clip(
                    cartoon_hsv[:, :, 1].astype(np.float32) * float(saturation_boost),
                    0,
                    255,
                ).astype(np.uint8)
            if value_boost != 1.0:
                cartoon_hsv[:, :, 2] = np.clip(
                    cartoon_hsv[:, :, 2].astype(np.float32) * float(value_boost),
                    0,
                    255,
                ).astype(np.uint8)
            cartoon = cv2.cvtColor(cartoon_hsv, cv2.COLOR_HSV2RGB)

        cartoon = _apply_gamma(cartoon, gamma)
        cartoon = _apply_local_contrast(cartoon, local_contrast)

        if face_mask is not None:
            mask_norm = (face_mask / 255.0).astype(np.float32)
            mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
            output = (cartoon.astype(np.float32) * (0.6 + 0.4 * mask_norm) +
                      img.astype(np.float32) * (0.4 * (1 - mask_norm)))
            cartoon = np.clip(output, 0, 255).astype(np.uint8)

        return cartoon

    @staticmethod
    def cartoon_effect_v2(img_rgb,
                        line_thickness=3,
                        canny_low=50,
                        canny_high=150,
                        bilateral_d=11,
                        bilateral_sigma_color=250,
                        median_blur_size=3,
                        contrast=1.1,
                        saturation_boost=1.2,
                        value_boost=1.0,
                        sharpen_amount=0.5,
                        tone_gamma=1.0,
                        local_contrast=0.0,
                        adaptive_canny_mix=0.45,
                        face_mask=None):
        """A more powerful and configurable cartoon effect algorithm."""
        img_rgb = _validate_rgb_image(img_rgb)
        img = img_rgb.copy()
        
        # 1. Advanced Edge Detection with Canny
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive_low, adaptive_high = _adaptive_canny(gray_blur, canny_low, canny_high, mix=adaptive_canny_mix)
        edges = cv2.Canny(gray_blur, adaptive_low, adaptive_high)
        
        edges_inv = cv2.bitwise_not(edges)
        
        # 2. Refine Line Work with Morphological Closing
        if line_thickness > 0:
            kernel = _get_structuring_element(line_thickness)
            if kernel is not None:
                edges_inv = cv2.morphologyEx(edges_inv, cv2.MORPH_CLOSE, kernel)
        
        # 3. Enhanced Color Quantization
        diameter, sigma_color, sigma_space = _resolve_bilateral_params(img.shape, bilateral_d)
        smooth = cv2.bilateralFilter(img, d=diameter, sigmaColor=max(bilateral_sigma_color, sigma_color), sigmaSpace=sigma_space)
        
        # 4. Apply a median blur to reduce color speckles
        if median_blur_size > 1:
            k = ensure_odd(median_blur_size)
            smooth = cv2.medianBlur(smooth, k)
            
        # 5. Combine Edges and Quantized Colors
        cartoon = cv2.bitwise_and(smooth, smooth, mask=edges_inv)
        
        # 6. Post-Processing for Vibrancy and Sharpness
        cartoon = np.clip(cartoon.astype(np.float32) * float(contrast), 0, 255).astype(np.uint8)
        if saturation_boost != 1.0 or value_boost != 1.0:
            hsv = cv2.cvtColor(cartoon, cv2.COLOR_RGB2HSV)
            if saturation_boost != 1.0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_boost, 0, 255).astype(np.uint8)
            if value_boost != 1.0:
                hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * value_boost, 0, 255).astype(np.uint8)
            cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        cartoon = _apply_gamma(cartoon, tone_gamma)
        cartoon = _apply_local_contrast(cartoon, local_contrast)
            
        if sharpen_amount > 0:
            blurred = cv2.GaussianBlur(cartoon, (0, 0), 3)
            cartoon = cv2.addWeighted(cartoon, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
            cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)

        # 7. Final Face Mask Blending (if provided)
        if face_mask is not None:
            mask_norm = (face_mask / 255.0).astype(np.float32)
            mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
            output = (cartoon.astype(np.float32) * (0.6 + 0.4 * mask_norm) +
                      img.astype(np.float32) * (0.4 * (1 - mask_norm)))
            cartoon = np.clip(output, 0, 255).astype(np.uint8)

        return cartoon

    @staticmethod
    def pencil_sketch(img_rgb, ksize=21, contrast=1.0):
        """Applies a pencil sketch effect to an RGB image."""
        img_rgb = _validate_rgb_image(img_rgb)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (ksize, ksize), 0)
        denom = 255 - blur
        denom[denom == 0] = 1
        sketch = cv2.divide(gray, denom, scale=256.0)
        sketch = np.clip(sketch * float(contrast), 0, 255).astype(np.uint8)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return sketch_rgb

    @staticmethod
    def watercolor(img_rgb, sigma_s=60, sigma_r=0.6):
        """Applies a watercolor painting effect to an RGB image."""
        img_rgb = _validate_rgb_image(img_rgb)
        try:
            styl = cv2.stylization(img_rgb, sigma_s=sigma_s, sigma_r=sigma_r)
            return styl
        except cv2.error as exc:
            logging.warning("OpenCV stylization unavailable, falling back to manual watercolor: %s", exc)
        except Exception as exc:
            logging.warning("Stylization failed unexpectedly: %s", exc)
        return _watercolor_fallback(img_rgb)

    @staticmethod
    def comic_dots(img_rgb, dot_scale=6):
        """Applies a comic book-style halftone dot effect."""
        img_rgb = _validate_rgb_image(img_rgb)
        h, w = img_rgb.shape[:2]
        small = cv2.resize(img_rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        small = cv2.bilateralFilter(small, 9, 250, 250)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        step = max(6, dot_scale)
        pad_h = int(math.ceil(h / step) * step)
        pad_w = int(math.ceil(w / step) * step)
        pad_bottom, pad_right = pad_h - h, pad_w - w
        padded_gray = cv2.copyMakeBorder(gray, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        coarse_h, coarse_w = pad_h // step, pad_w // step
        samples = cv2.resize(padded_gray, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
        samples = samples.astype(np.float32) / 255.0
        radius_map = np.clip((1.0 - samples) * (step / 2.0), 1.0, step / 2.0)

        dist = _prepare_halftone_distances(step)
        mask_tiles = (dist <= radius_map[:, :, None, None]).astype(np.uint8)
        mask = mask_tiles.reshape(coarse_h, coarse_w, step, step)
        mask = mask.swapaxes(1, 2).reshape(pad_h, pad_w)
        mask = mask[:h, :w]
        dot_mask = (mask * 255).astype(np.uint8)

        blended = cv2.bitwise_and(up, up, mask=255 - dot_mask)
        edges = cv2.Canny(gray, 100, 200)
        edges_inv = cv2.bitwise_not(edges)
        edges_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)
        out = cv2.bitwise_and(blended, edges_col)
        return out