"""
Character shape vector generation.

Renders each ASCII character and samples it with 6 circles to create
a 6D shape vector that captures where the character's visual density lies.
"""

import math
import string
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SamplingCircle:
    """A circle used for sampling character density."""
    cx: float  # center x (0-1 normalized)
    cy: float  # center y (0-1 normalized)
    radius: float  # radius (0-1 normalized)


# 6 sampling circles in a 2x3 staggered grid
# Staggering reduces gaps between circles for better coverage
INTERNAL_CIRCLES = [
    # Upper row (left slightly lower, right slightly higher)
    SamplingCircle(0.30, 0.20, 0.22),
    SamplingCircle(0.70, 0.15, 0.22),
    # Middle row
    SamplingCircle(0.30, 0.50, 0.22),
    SamplingCircle(0.70, 0.50, 0.22),
    # Lower row (left slightly higher, right slightly lower)
    SamplingCircle(0.30, 0.80, 0.22),
    SamplingCircle(0.70, 0.85, 0.22),
]

# 10 external sampling circles for directional contrast enhancement
# These reach outside the cell boundary to detect neighboring regions
EXTERNAL_CIRCLES = [
    # Top edge
    SamplingCircle(0.30, -0.15, 0.18),
    SamplingCircle(0.70, -0.15, 0.18),
    # Left edge
    SamplingCircle(-0.15, 0.25, 0.18),
    SamplingCircle(-0.15, 0.75, 0.18),
    # Right edge
    SamplingCircle(1.15, 0.25, 0.18),
    SamplingCircle(1.15, 0.75, 0.18),
    # Bottom edge
    SamplingCircle(0.30, 1.15, 0.18),
    SamplingCircle(0.70, 1.15, 0.18),
    # Corners (for diagonal boundaries)
    SamplingCircle(-0.10, -0.10, 0.15),
    SamplingCircle(1.10, 1.10, 0.15),
]

# Mapping from internal circle index to external circles that affect it
# Used for directional contrast enhancement
AFFECTING_EXTERNAL = [
    [0, 1, 2, 8],      # top-left internal <- top + left + corner
    [0, 1, 4, 8],      # top-right internal <- top + right
    [2, 3],            # middle-left internal <- left
    [4, 5],            # middle-right internal <- right
    [3, 6, 7, 9],      # bottom-left internal <- left + bottom
    [5, 6, 7, 9],      # bottom-right internal <- right + bottom + corner
]


def get_monospace_font(size: int = 64) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a monospace font for rendering characters."""
    # Try common monospace fonts
    font_names = [
        "DejaVuSansMono.ttf",
        "Menlo.ttc",
        "Monaco.ttf",
        "Consolas.ttf",
        "LiberationMono-Regular.ttf",
        "Courier New.ttf",
        "monospace",
    ]

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue

    # Fallback to default font
    return ImageFont.load_default()


def render_character(char: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
                     width: int, height: int) -> np.ndarray:
    """
    Render a character to a grayscale numpy array.

    Returns array of shape (height, width) with values 0.0-1.0,
    where 1.0 is white (character ink) and 0.0 is black (background).
    """
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # Get character bounding box to center it
    bbox = draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # Center the character in the cell
    x = (width - char_width) // 2 - bbox[0]
    y = (height - char_height) // 2 - bbox[1]

    draw.text((x, y), char, fill=255, font=font)

    return np.array(img, dtype=np.float32) / 255.0


def sample_circle(image: np.ndarray, circle: SamplingCircle,
                  samples_per_circle: int = 32) -> float:
    """
    Sample a circle region and return average intensity.

    Uses stratified sampling within the circle for better coverage.
    """
    height, width = image.shape
    cx = circle.cx * width
    cy = circle.cy * height
    radius = circle.radius * min(width, height)

    total = 0.0
    count = 0

    # Sample in a grid pattern within the circle's bounding box
    samples_per_dim = int(math.sqrt(samples_per_circle))
    for i in range(samples_per_dim):
        for j in range(samples_per_dim):
            # Offset within the circle's bounding box
            fx = (i + 0.5) / samples_per_dim * 2 - 1  # -1 to 1
            fy = (j + 0.5) / samples_per_dim * 2 - 1

            # Skip if outside circle
            if fx * fx + fy * fy > 1:
                continue

            # Convert to image coordinates
            px = int(cx + fx * radius)
            py = int(cy + fy * radius)

            # Skip if outside image bounds
            if 0 <= px < width and 0 <= py < height:
                total += image[py, px]
                count += 1

    return total / count if count > 0 else 0.0


def compute_shape_vector(image: np.ndarray,
                         circles: list[SamplingCircle],
                         samples_per_circle: int = 32) -> np.ndarray:
    """Compute a shape vector by sampling multiple circles."""
    return np.array([
        sample_circle(image, circle, samples_per_circle)
        for circle in circles
    ])


@dataclass
class CharacterShape:
    """A character and its associated shape vector."""
    char: str
    vector: np.ndarray


def generate_character_shapes(
    charset: str | None = None,
    cell_width: int = 10,
    cell_height: int = 18,
    font_size: int = 64,
    samples_per_circle: int = 64,
) -> list[CharacterShape]:
    """
    Generate shape vectors for all characters in the charset.

    Args:
        charset: Characters to include. Defaults to printable ASCII.
        cell_width: Width of rendering cell (scaled for sampling).
        cell_height: Height of rendering cell (scaled for sampling).
        font_size: Font size for rendering (larger = more accurate).
        samples_per_circle: Number of samples per sampling circle.

    Returns:
        List of CharacterShape objects with normalized shape vectors.
    """
    if charset is None:
        # Printable ASCII excluding space (we'll handle space specially)
        charset = string.printable[:95]

    font = get_monospace_font(font_size)

    # Render at higher resolution for better sampling accuracy
    render_width = cell_width * 8
    render_height = cell_height * 8

    shapes = []
    for char in charset:
        if char in '\t\n\r\x0b\x0c':  # Skip whitespace control chars
            continue

        image = render_character(char, font, render_width, render_height)
        vector = compute_shape_vector(image, INTERNAL_CIRCLES, samples_per_circle)
        shapes.append(CharacterShape(char, vector))

    # Normalize vectors: for each dimension, divide by max across all chars
    if shapes:
        all_vectors = np.array([s.vector for s in shapes])
        max_per_dim = np.maximum(all_vectors.max(axis=0), 1e-6)  # Avoid div by 0

        for shape in shapes:
            shape.vector = shape.vector / max_per_dim

    return shapes


# Cache for precomputed character shapes
_shape_cache: dict[tuple, list[CharacterShape]] = {}


def get_character_shapes(
    charset: str | None = None,
    cell_width: int = 10,
    cell_height: int = 18,
) -> list[CharacterShape]:
    """Get character shapes, using cache for repeated calls."""
    key = (charset, cell_width, cell_height)
    if key not in _shape_cache:
        _shape_cache[key] = generate_character_shapes(
            charset=charset,
            cell_width=cell_width,
            cell_height=cell_height,
        )
    return _shape_cache[key]
