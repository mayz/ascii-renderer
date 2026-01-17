"""
Contrast enhancement algorithms for sharper ASCII edges.

Two types of enhancement:
1. Global: Enhances contrast within a cell's sampling vector
2. Directional: Uses external samples to enhance boundaries with neighbors
"""

import numpy as np

from .shape import AFFECTING_EXTERNAL


def apply_global_contrast(
    vector: np.ndarray,
    exponent: float = 1.0,
) -> np.ndarray:
    """
    Apply global contrast enhancement to a sampling vector.

    Normalizes the vector to its max value, applies an exponent to
    increase contrast, then denormalizes back.

    Args:
        vector: The 6D sampling vector (values 0-1).
        exponent: Power to raise normalized values to. Higher = more contrast.

    Returns:
        Enhanced sampling vector.
    """
    if exponent <= 1.0:
        return vector.copy()

    max_val = vector.max()
    if max_val < 1e-6:
        return vector.copy()

    # Normalize, apply exponent, denormalize
    normalized = vector / max_val
    enhanced = np.power(normalized, exponent)
    return enhanced * max_val


def apply_directional_contrast(
    internal: np.ndarray,
    external: np.ndarray,
    exponent: float = 1.0,
) -> np.ndarray:
    """
    Apply directional contrast enhancement using external samples.

    For each internal component, finds the max among affecting external
    components, normalizes to that max, applies exponent, denormalizes.

    This sharpens boundaries between the cell and its neighbors.

    Args:
        internal: The 6D internal sampling vector.
        external: The 10D external sampling vector.
        exponent: Power to raise normalized values to. Higher = more contrast.

    Returns:
        Enhanced sampling vector.
    """
    if exponent <= 1.0:
        return internal.copy()

    enhanced = internal.copy()

    for i in range(len(internal)):
        # Find max among this component and its affecting external components
        affecting_indices = AFFECTING_EXTERNAL[i]
        affecting_values = external[affecting_indices]
        max_external = affecting_values.max() if len(affecting_values) > 0 else 0.0
        max_val = max(internal[i], max_external)

        if max_val < 1e-6:
            continue

        # Normalize, apply exponent, denormalize
        normalized = internal[i] / max_val
        enhanced[i] = (normalized ** exponent) * max_val

    return enhanced


def enhance_sampling_vector(
    internal: np.ndarray,
    external: np.ndarray | None = None,
    global_exponent: float = 1.0,
    directional_exponent: float = 1.0,
) -> np.ndarray:
    """
    Apply both contrast enhancements to a sampling vector.

    Directional enhancement is applied first (if external samples provided),
    then global enhancement.

    Args:
        internal: The 6D internal sampling vector.
        external: The 10D external sampling vector (optional).
        global_exponent: Exponent for global contrast enhancement.
        directional_exponent: Exponent for directional contrast enhancement.

    Returns:
        Enhanced sampling vector.
    """
    result = internal.copy()

    # Apply directional first (requires external samples)
    if external is not None and directional_exponent > 1.0:
        result = apply_directional_contrast(result, external, directional_exponent)

    # Apply global second
    if global_exponent > 1.0:
        result = apply_global_contrast(result, global_exponent)

    return result
