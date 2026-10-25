"""Seeded grayscale scenes with object masks and distinct position definitions."""

from dataclasses import dataclass

import numpy as np

from utils.functions.centroid_diagnostics import pixel_mean

CASES = (
    "circle",
    "ellipse",
    "asymmetric",
    "separated",
    "larger_brighter",
    "approach",
    "crossing",
    "partial_overlap",
    "occlusion",
    "ring",
    "background_rise",
    "brightness_change",
    "noise",
    "border",
    "missing",
    "gap",
    "reversal",
    "stop",
    "radius_change",
)


@dataclass
class ImageSequence:
    images: np.ndarray
    masks: np.ndarray
    centers: np.ndarray
    isolated_centroids: np.ndarray
    observable: np.ndarray
    source_frames: np.ndarray
    times: np.ndarray
    segments: np.ndarray
    object_ids: tuple = ("target", "distractor")


def generate_images(case, seed, count=24, size=96):
    """Render uint16 TIFF-like scenes; observability is conservative by design."""
    if case not in CASES or count < 8 or size < 64:
        raise ValueError("Unknown case, too few frames, or too small a canvas")
    rng = np.random.default_rng(seed)
    times = np.arange(count) / 200
    segments = np.zeros(count, dtype=int)
    if case == "gap":
        times[count // 2 :] += 1
        segments[count // 2 :] = 1
    angle = np.arange(count) * 0.35
    if case == "reversal":
        angle[count // 2 :] = angle[count // 2] - np.arange(count - count // 2) * 0.35
    if case == "stop":
        start, end = count // 3, 2 * count // 3
        angle[start:end] = angle[start]
        angle[end:] -= (end - start) * 0.35
    radius = np.full(count, 12.0)
    if case == "radius_change":
        radius = np.linspace(7, 18, count)
    centers = np.full((count, 2, 2), np.nan)
    origin = size / 2 + rng.uniform(-2, 2, size=2)
    centers[:, 0, 0] = origin[0] + radius * np.cos(angle)
    centers[:, 0, 1] = origin[1] + radius * np.sin(angle)
    if case == "border":
        centers[:, 0, 0] = 2 + 3 * np.sin(angle)
    masks = np.zeros((count, 2, size, size), dtype=bool)
    images = np.empty((count, size, size), dtype=np.uint16)
    observable = np.ones(count, dtype=bool)
    y, x = np.mgrid[:size, :size]
    multiple = case in {"separated", "larger_brighter", "approach", "crossing", "partial_overlap", "occlusion"}
    for index in range(count):
        cx, cy = centers[index, 0]
        a, b = (9, 5) if case == "ellipse" else (7, 7)
        shape = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
        if case == "ring":
            shape &= (x - cx) ** 2 + (y - cy) ** 2 >= 4**2
        if case == "asymmetric":
            shape &= ~((x > cx) & (y < cy))
        if case == "missing" and count // 3 <= index < 2 * count // 3:
            shape[:] = False
            observable[index] = False
        masks[index, 0] = shape
        if multiple:
            offset = 25.0
            if case in {"approach", "crossing"}:
                offset = 30 - 60 * index / (count - 1)
            elif case == "partial_overlap":
                offset = 9
            elif case == "occlusion":
                offset = 0 if count // 3 <= index < 2 * count // 3 else 25
            centers[index, 1] = (cx + offset, cy)
            r = 11 if case in {"larger_brighter", "occlusion"} else 7
            masks[index, 1] = (x - cx - offset) ** 2 + (y - cy) ** 2 <= r**2
            # Hidden target pixels prevent recovering the isolated-mask centroid
            # without a shape prior. Mere adjacency does not hide target pixels.
            observable[index] &= not np.any(shape & masks[index, 1])
        background = 1500 + (index * 200 if case == "background_rise" else 0)
        image = np.full((size, size), background, dtype=float)
        intensity = 17000 - index * 500 if case == "brightness_change" else 17000
        image[masks[index, 0]] = intensity
        image[masks[index, 1]] = 26000  # Foreground distractor occludes target.
        if case == "noise":
            image += rng.normal(0, 1800, image.shape)
        images[index] = np.clip(image, 0, 65535).astype(np.uint16)
    centroids = np.array([[pixel_mean(mask) for mask in frame] for frame in masks])
    return ImageSequence(images, masks, centers, centroids, observable, np.arange(1, count + 1), times, segments)
