"""X-ray bias-normalization preprocessing.

Goal: strip machine/scanner-specific signal so inference does not depend on
which X-ray machine produced the image. Pipeline order:

    grayscale -> polarity correction -> hand crop -> CLAHE -> square pad -> resize

- polarity: some detectors store bone-dark/background-bright; we standardize to
  bone-bright/background-dark.
- hand crop: removes collimation borders, lead markers, and background — the
  pixels that most strongly fingerprint a specific machine/room.
- CLAHE: adaptive contrast equalization normalizes global brightness/contrast
  differences between detectors.
- square pad before resize: avoids the aspect-ratio distortion of a direct
  Resize((S, S)) on non-square X-rays.
"""

import cv2
import numpy as np
from PIL import Image
import config


def _to_gray_array(img) -> np.ndarray:
    """Accept a path, PIL.Image, or ndarray; return uint8 grayscale."""
    if isinstance(img, (str,)) or hasattr(img, "__fspath__"):
        arr = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if arr is None:  # fallback for encodings cv2 can't read
            arr = np.array(Image.open(img).convert("L"))
    elif isinstance(img, Image.Image):
        arr = np.array(img.convert("L"))
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def correct_polarity(gray: np.ndarray) -> np.ndarray:
    """Invert if the border is brighter than the center (inverted polarity)."""
    h, w = gray.shape
    bh, bw = max(1, h // 10), max(1, w // 10)
    border = np.concatenate([
        gray[:bh, :].ravel(), gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel(),
    ])
    center = gray[h // 2 - h // 6: h // 2 + h // 6, w // 2 - w // 6: w // 2 + w // 6]
    if border.mean() > center.mean():
        gray = 255 - gray
    return gray


def crop_to_hand(gray: np.ndarray, pad_frac: float = 0.04) -> np.ndarray:
    """Crop to the hand's bounding box. Falls back to the full frame on failure."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * gray.size:  # segmentation likely failed
        return gray

    x, y, w, h = cv2.boundingRect(c)
    pad = int(pad_frac * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return gray[y0:y1, x0:x1]


def apply_clahe(gray: np.ndarray, clip: float = None, grid: int = None) -> np.ndarray:
    clip = config.CLAHE_CLIP if clip is None else clip
    grid = config.CLAHE_GRID if grid is None else grid
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def pad_to_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(gray, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=0)


def preprocess_xray(img, out_size: int | None = None) -> Image.Image:
    """Run the full bias-normalization pipeline. Returns an RGB PIL.Image."""
    gray = _to_gray_array(img)
    gray = correct_polarity(gray)
    gray = crop_to_hand(gray)
    gray = apply_clahe(gray)
    gray = pad_to_square(gray)
    if out_size:
        gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
