"""
Image processing and ASCII conversion.

Converts images to ASCII by:
1. Dividing the image into a grid of cells
2. Sampling each cell with 6 circles to get shape vectors
3. Optionally sampling external circles for edge enhancement
4. Applying contrast enhancement
5. Looking up the best matching character for each cell
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .contrast import enhance_sampling_vector
from .lookup import CharacterLookup, get_lookup
from .shape import (
    EXTERNAL_CIRCLES,
    INTERNAL_CIRCLES,
    SamplingCircle,
)


def rgb_to_luminance(r: float, g: float, b: float) -> float:
    """Convert RGB to relative luminance (0-1)."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def load_image_as_luminance(path: str) -> np.ndarray:
    """
    Load an image and convert to luminance array.

    Returns array of shape (height, width) with values 0.0-1.0.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Apply luminance formula
    luminance = (
        0.2126 * arr[:, :, 0] +
        0.7152 * arr[:, :, 1] +
        0.0722 * arr[:, :, 2]
    )
    return luminance


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image and return both luminance and RGB arrays.

    Returns:
        Tuple of (luminance, rgb) where:
        - luminance: shape (height, width), values 0.0-1.0
        - rgb: shape (height, width, 3), values 0.0-1.0
    """
    img = Image.open(path).convert("RGB")
    rgb = np.array(img, dtype=np.float32) / 255.0

    # Calculate luminance
    luminance = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )
    return luminance, rgb


def sample_cell_color(
    rgb: np.ndarray,
    cell_x: int,
    cell_y: int,
    cell_width: float,
    cell_height: float,
) -> tuple[int, int, int]:
    """
    Sample the average color of a cell.

    Args:
        rgb: RGB array of shape (height, width, 3), values 0.0-1.0.
        cell_x: Cell column index.
        cell_y: Cell row index.
        cell_width: Width of each cell in pixels.
        cell_height: Height of each cell in pixels.

    Returns:
        Tuple of (r, g, b) with values 0-255.
    """
    img_height, img_width = rgb.shape[:2]

    # Calculate cell bounds
    x0 = int(cell_x * cell_width)
    x1 = int((cell_x + 1) * cell_width)
    y0 = int(cell_y * cell_height)
    y1 = int((cell_y + 1) * cell_height)

    # Clamp to image bounds
    x0 = max(0, min(x0, img_width - 1))
    x1 = max(x0 + 1, min(x1, img_width))
    y0 = max(0, min(y0, img_height - 1))
    y1 = max(y0 + 1, min(y1, img_height))

    # Extract cell region and compute average color
    cell_rgb = rgb[y0:y1, x0:x1]
    avg_color = cell_rgb.mean(axis=(0, 1))

    # Convert to 0-255 range
    return (
        int(avg_color[0] * 255),
        int(avg_color[1] * 255),
        int(avg_color[2] * 255),
    )


def sample_circle_from_image(
    image: np.ndarray,
    circle: SamplingCircle,
    cell_x: int,
    cell_y: int,
    cell_width: float,
    cell_height: float,
    samples_per_dim: int = 4,
) -> float:
    """
    Sample a circle region from the image at a specific cell location.

    Args:
        image: Luminance array (height, width).
        circle: Sampling circle (normalized 0-1 coordinates).
        cell_x: Cell column index.
        cell_y: Cell row index.
        cell_width: Width of each cell in pixels.
        cell_height: Height of each cell in pixels.
        samples_per_dim: Samples per dimension within the circle.

    Returns:
        Average luminance within the circle (0-1).
    """
    img_height, img_width = image.shape

    # Convert circle to image coordinates
    cx = (cell_x + circle.cx) * cell_width
    cy = (cell_y + circle.cy) * cell_height
    radius = circle.radius * min(cell_width, cell_height)

    total = 0.0
    count = 0

    # Sample in a grid pattern within the circle
    for i in range(samples_per_dim):
        for j in range(samples_per_dim):
            fx = (i + 0.5) / samples_per_dim * 2 - 1  # -1 to 1
            fy = (j + 0.5) / samples_per_dim * 2 - 1

            # Skip if outside circle
            if fx * fx + fy * fy > 1:
                continue

            # Convert to image coordinates
            px = int(cx + fx * radius)
            py = int(cy + fy * radius)

            # Clamp to image bounds and sample
            px = max(0, min(px, img_width - 1))
            py = max(0, min(py, img_height - 1))

            total += image[py, px]
            count += 1

    return total / count if count > 0 else 0.0


def sample_cell(
    image: np.ndarray,
    cell_x: int,
    cell_y: int,
    cell_width: float,
    cell_height: float,
    sample_quality: int = 16,
    include_external: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sample internal and external circles for a single cell.

    Args:
        image: Luminance array.
        cell_x: Cell column index.
        cell_y: Cell row index.
        cell_width: Width of each cell in pixels.
        cell_height: Height of each cell in pixels.
        sample_quality: Total samples per circle (roughly).
        include_external: Whether to sample external circles.

    Returns:
        Tuple of (internal_vector, external_vector).
        external_vector is None if include_external is False.
    """
    samples_per_dim = max(2, int(math.sqrt(sample_quality)))

    # Sample internal circles
    internal = np.array([
        sample_circle_from_image(
            image, circle, cell_x, cell_y,
            cell_width, cell_height, samples_per_dim
        )
        for circle in INTERNAL_CIRCLES
    ])

    # Sample external circles if requested
    external = None
    if include_external:
        external = np.array([
            sample_circle_from_image(
                image, circle, cell_x, cell_y,
                cell_width, cell_height, samples_per_dim
            )
            for circle in EXTERNAL_CIRCLES
        ])

    return internal, external


@dataclass
class RenderConfig:
    """Configuration for ASCII rendering."""
    width: int = 80  # Output width in characters
    cell_aspect: float = 0.5  # Character width/height ratio (monospace chars are taller)
    global_contrast: float = 1.0  # Global contrast enhancement exponent
    edge_contrast: float = 1.0  # Directional edge contrast exponent
    sample_quality: int = 16  # Samples per circle
    charset: str | None = None  # Character set (None = printable ASCII)
    invert: bool = False  # Invert for dark backgrounds
    color: bool = False  # Enable colored output (24-bit ANSI true color)


def render_to_ascii(
    image: np.ndarray,
    config: RenderConfig | None = None,
    lookup: CharacterLookup | None = None,
    rgb: np.ndarray | None = None,
) -> str:
    """
    Convert a luminance image to ASCII art.

    Args:
        image: Luminance array of shape (height, width), values 0-1.
        config: Rendering configuration.
        lookup: Character lookup table (created if not provided).
        rgb: Optional RGB array for colored output, shape (height, width, 3).

    Returns:
        ASCII art string with newlines (and ANSI codes if color enabled).
    """
    from .color import colorize_line

    if config is None:
        config = RenderConfig()

    img_height, img_width = image.shape

    # Calculate grid dimensions
    num_cols = config.width
    cell_width = img_width / num_cols
    cell_height = cell_width / config.cell_aspect
    num_rows = int(img_height / cell_height)

    if num_rows < 1:
        num_rows = 1
        cell_height = img_height

    # Get or create lookup table
    if lookup is None:
        lookup = get_lookup(
            charset=config.charset,
            cell_width=int(cell_width),
            cell_height=int(cell_height),
        )

    # Invert image if needed (for dark terminal backgrounds)
    if config.invert:
        image = 1.0 - image

    # Determine if we need external samples
    need_external = config.edge_contrast > 1.0

    # Check if we can do color output
    use_color = config.color and rgb is not None

    # Process each cell
    lines = []
    for row in range(num_rows):
        chars = []
        colors = [] if use_color else None

        for col in range(num_cols):
            # Sample the cell
            internal, external = sample_cell(
                image, col, row,
                cell_width, cell_height,
                config.sample_quality,
                include_external=need_external,
            )

            # Apply contrast enhancement
            enhanced = enhance_sampling_vector(
                internal,
                external,
                global_exponent=config.global_contrast,
                directional_exponent=config.edge_contrast,
            )

            # Find best matching character
            char = lookup.find_best(enhanced)
            chars.append(char)

            # Sample color if needed
            if use_color:
                color = sample_cell_color(rgb, col, row, cell_width, cell_height)
                colors.append(color)

        # Build line with or without color
        if use_color:
            lines.append(colorize_line(chars, colors))
        else:
            lines.append("".join(chars))

    return "\n".join(lines)


def render_file_to_ascii(
    path: str,
    config: RenderConfig | None = None,
) -> str:
    """
    Load an image file and convert to ASCII art.

    Args:
        path: Path to image file.
        config: Rendering configuration.

    Returns:
        ASCII art string.
    """
    if config is None:
        config = RenderConfig()

    if config.color:
        # Load both luminance and RGB for colored output
        luminance, rgb = load_image(path)
        return render_to_ascii(luminance, config, rgb=rgb)
    else:
        # Just load luminance for plain output
        image = load_image_as_luminance(path)
        return render_to_ascii(image, config)
