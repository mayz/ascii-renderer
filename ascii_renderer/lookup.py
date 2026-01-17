"""
Nearest neighbor character lookup in shape vector space.

Finds the ASCII character whose shape vector best matches
a given sampling vector from the image.
"""

import numpy as np

from .shape import CharacterShape, get_character_shapes


class CharacterLookup:
    """
    Lookup table for finding best-matching ASCII characters.

    Uses Euclidean distance in 6D shape vector space.
    """

    def __init__(
        self,
        charset: str | None = None,
        cell_width: int = 10,
        cell_height: int = 18,
        use_kdtree: bool = True,
    ):
        """
        Initialize the lookup table.

        Args:
            charset: Characters to use. Defaults to printable ASCII.
            cell_width: Character cell width for shape generation.
            cell_height: Character cell height for shape generation.
            use_kdtree: Use scipy KDTree for faster lookups if available.
        """
        self.shapes = get_character_shapes(charset, cell_width, cell_height)
        self.chars = [s.char for s in self.shapes]
        self.vectors = np.array([s.vector for s in self.shapes])

        self._kdtree = None
        if use_kdtree:
            try:
                from scipy.spatial import KDTree
                self._kdtree = KDTree(self.vectors)
            except ImportError:
                pass  # Fall back to brute force

    def find_best(self, sampling_vector: np.ndarray) -> str:
        """
        Find the character whose shape best matches the sampling vector.

        Args:
            sampling_vector: 6D vector sampled from the image cell.

        Returns:
            The best-matching ASCII character.
        """
        if self._kdtree is not None:
            _, idx = self._kdtree.query(sampling_vector)
            return self.chars[idx]

        # Brute force: compute all distances
        distances = np.sum((self.vectors - sampling_vector) ** 2, axis=1)
        return self.chars[np.argmin(distances)]

    def find_best_batch(self, sampling_vectors: np.ndarray) -> list[str]:
        """
        Find best characters for multiple sampling vectors.

        Args:
            sampling_vectors: Array of shape (N, 6) with N sampling vectors.

        Returns:
            List of N best-matching characters.
        """
        if self._kdtree is not None:
            _, indices = self._kdtree.query(sampling_vectors)
            return [self.chars[i] for i in indices]

        # Brute force batch
        results = []
        for vec in sampling_vectors:
            distances = np.sum((self.vectors - vec) ** 2, axis=1)
            results.append(self.chars[np.argmin(distances)])
        return results


# Module-level cache for lookup instances
_lookup_cache: dict[tuple, CharacterLookup] = {}


def get_lookup(
    charset: str | None = None,
    cell_width: int = 10,
    cell_height: int = 18,
) -> CharacterLookup:
    """Get a cached CharacterLookup instance."""
    key = (charset, cell_width, cell_height)
    if key not in _lookup_cache:
        _lookup_cache[key] = CharacterLookup(charset, cell_width, cell_height)
    return _lookup_cache[key]
