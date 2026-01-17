"""Tests for core ASCII renderer functionality."""

import numpy as np
import pytest

from ascii_renderer.contrast import (
    apply_directional_contrast,
    apply_global_contrast,
    enhance_sampling_vector,
)
from ascii_renderer.lookup import CharacterLookup
from ascii_renderer.processor import RenderConfig, render_to_ascii
from ascii_renderer.shape import (
    AFFECTING_EXTERNAL,
    INTERNAL_CIRCLES,
    CharacterShape,
    compute_shape_vector,
    generate_character_shapes,
)


class TestShapeVectors:
    """Tests for character shape vector generation."""

    def test_internal_circles_count(self):
        """Should have exactly 6 internal sampling circles."""
        assert len(INTERNAL_CIRCLES) == 6

    def test_affecting_external_mapping(self):
        """Each internal circle should map to some external circles."""
        assert len(AFFECTING_EXTERNAL) == 6
        for indices in AFFECTING_EXTERNAL:
            assert len(indices) > 0
            assert all(0 <= i < 10 for i in indices)

    def test_generate_character_shapes_returns_list(self):
        """Should return a list of CharacterShape objects."""
        shapes = generate_character_shapes(charset="ABC")
        assert isinstance(shapes, list)
        assert len(shapes) == 3
        assert all(isinstance(s, CharacterShape) for s in shapes)

    def test_shape_vectors_are_normalized(self):
        """Shape vectors should be normalized (max ~1.0 per dimension)."""
        # Use a diverse charset
        shapes = generate_character_shapes(charset="@MWN#XO^_|-.,'`")
        vectors = np.array([s.vector for s in shapes])

        # Max per dimension should be 1.0 (that's how normalization works)
        max_per_dim = vectors.max(axis=0)
        # Dimensions with data should be normalized to 1.0
        nonzero_dims = max_per_dim > 0
        if nonzero_dims.any():
            assert np.allclose(max_per_dim[nonzero_dims], 1.0, atol=0.01), \
                f"Non-zero dims should be ~1.0: {max_per_dim}"

    def test_shape_vectors_are_6d(self):
        """Each shape vector should be 6-dimensional."""
        shapes = generate_character_shapes(charset="X")
        assert shapes[0].vector.shape == (6,)

    def test_compute_shape_vector_white_image(self):
        """White image should give high values."""
        white = np.ones((100, 60), dtype=np.float32)
        vector = compute_shape_vector(white, INTERNAL_CIRCLES)
        assert all(v > 0.9 for v in vector)

    def test_compute_shape_vector_black_image(self):
        """Black image should give low values."""
        black = np.zeros((100, 60), dtype=np.float32)
        vector = compute_shape_vector(black, INTERNAL_CIRCLES)
        assert all(v < 0.1 for v in vector)


class TestContrastEnhancement:
    """Tests for contrast enhancement algorithms."""

    def test_global_contrast_no_change_when_exponent_1(self):
        """Exponent 1.0 should not change the vector."""
        vec = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        result = apply_global_contrast(vec, exponent=1.0)
        np.testing.assert_array_almost_equal(result, vec)

    def test_global_contrast_preserves_max(self):
        """Max value should be preserved after enhancement."""
        vec = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        result = apply_global_contrast(vec, exponent=2.0)
        assert np.isclose(result.max(), vec.max())

    def test_global_contrast_increases_spread(self):
        """Enhancement should increase spread between values."""
        vec = np.array([0.8, 0.6, 0.4])
        result = apply_global_contrast(vec, exponent=3.0)

        # Ratio between max and min should increase
        original_ratio = vec.max() / (vec.min() + 1e-6)
        enhanced_ratio = result.max() / (result.min() + 1e-6)
        assert enhanced_ratio > original_ratio

    def test_directional_contrast_no_change_when_exponent_1(self):
        """Exponent 1.0 should not change the vector."""
        internal = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        external = np.array([0.6, 0.4, 0.5, 0.3, 0.7, 0.5, 0.4, 0.3, 0.2, 0.6])
        result = apply_directional_contrast(internal, external, exponent=1.0)
        np.testing.assert_array_almost_equal(result, internal)

    def test_directional_contrast_darker_when_neighbor_brighter(self):
        """Component should get darker when external neighbor is brighter."""
        internal = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # Make only external indices 0, 1, 8 bright (these affect internal 0 and 1 only)
        # External index 2 is shared between internal 0 and 2, so avoid it
        external = np.zeros(10)
        external[0] = 0.9  # affects internal 0, 1
        external[1] = 0.9  # affects internal 0, 1
        external[8] = 0.9  # affects internal 0, 1

        result = apply_directional_contrast(internal, external, exponent=3.0)
        # Internal indices 0, 1 should be darker than indices 2, 3, 4, 5
        assert result[0] < result[2], f"result[0]={result[0]}, result[2]={result[2]}"
        assert result[0] < result[3]
        assert result[1] < result[4]

    def test_enhance_sampling_vector_combines_both(self):
        """Should apply both directional and global enhancement."""
        internal = np.array([0.6, 0.4, 0.5, 0.3, 0.7, 0.5])
        external = np.array([0.8, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.7, 0.4])

        result = enhance_sampling_vector(
            internal, external,
            global_exponent=2.0,
            directional_exponent=2.0,
        )

        # Should be different from input
        assert not np.allclose(result, internal)
        # Should still be in valid range
        assert all(0 <= v <= 1 for v in result)


class TestCharacterLookup:
    """Tests for character lookup functionality."""

    def test_lookup_returns_character(self):
        """Lookup should return a single character string."""
        lookup = CharacterLookup(charset="@.|-+")
        vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = lookup.find_best(vec)
        assert isinstance(result, str)
        assert len(result) == 1

    def test_lookup_high_density_returns_dense_char(self):
        """High-density vector should return a dense character like @."""
        lookup = CharacterLookup(charset="@. ")
        vec = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = lookup.find_best(vec)
        assert result == "@"

    def test_lookup_low_density_returns_sparse_char(self):
        """Low-density vector should return space or dot."""
        lookup = CharacterLookup(charset="@. ")
        vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = lookup.find_best(vec)
        assert result in ". "

    def test_batch_lookup(self):
        """Batch lookup should return list of characters."""
        lookup = CharacterLookup(charset="@.|-+")
        vecs = np.array([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ])
        results = lookup.find_best_batch(vecs)
        assert len(results) == 2
        assert all(isinstance(c, str) for c in results)


class TestProcessor:
    """Tests for image processing and ASCII conversion."""

    def test_render_white_image(self):
        """White image should produce dense characters."""
        white = np.ones((100, 200), dtype=np.float32)
        config = RenderConfig(width=20, charset="@. ")
        result = render_to_ascii(white, config)

        # Should have mostly @ characters
        assert "@" in result
        lines = result.strip().split("\n")
        assert len(lines) > 0

    def test_render_black_image(self):
        """Black image should produce sparse characters."""
        black = np.zeros((100, 200), dtype=np.float32)
        config = RenderConfig(width=20, charset="@. ")
        result = render_to_ascii(black, config)

        # Should have mostly spaces or dots
        lines = result.strip().split("\n")
        assert len(lines) > 0
        # Dense chars should be rare
        assert result.count("@") < result.count(" ") + result.count(".")

    def test_render_respects_width(self):
        """Output width should match config."""
        image = np.random.rand(100, 200).astype(np.float32)
        config = RenderConfig(width=40)
        result = render_to_ascii(image, config)

        lines = result.strip().split("\n")
        assert all(len(line) == 40 for line in lines)

    def test_render_invert_swaps_density(self):
        """Invert option should swap light and dark."""
        gradient = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)

        normal = render_to_ascii(gradient, RenderConfig(width=10, charset="@."))
        inverted = render_to_ascii(gradient, RenderConfig(width=10, charset="@.", invert=True))

        # They should be different
        assert normal != inverted

    def test_render_with_contrast_enhancement(self):
        """Contrast enhancement should produce valid output."""
        np.random.seed(42)  # Fixed seed for reproducibility
        image = np.random.rand(100, 200).astype(np.float32)
        config = RenderConfig(
            width=20,
            global_contrast=2.5,
            edge_contrast=3.0,
        )
        result = render_to_ascii(image, config)

        lines = result.split("\n")
        assert len(lines) > 0
        assert all(len(line) == 20 for line in lines)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_image(self):
        """Should handle very small images."""
        tiny = np.array([[0.5]], dtype=np.float32)
        config = RenderConfig(width=1)
        result = render_to_ascii(tiny, config)
        assert len(result.strip()) >= 1

    def test_uniform_image(self):
        """Should handle uniform images without crashing."""
        uniform = np.full((50, 100), 0.5, dtype=np.float32)
        config = RenderConfig(width=20)
        result = render_to_ascii(uniform, config)
        assert len(result) > 0

    def test_empty_charset_raises(self):
        """Empty charset should raise or handle gracefully."""
        with pytest.raises(Exception):
            CharacterLookup(charset="")

    def test_single_char_charset(self):
        """Single character charset should work."""
        lookup = CharacterLookup(charset="X")
        vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = lookup.find_best(vec)
        assert result == "X"


class TestColor:
    """Tests for color output functionality."""

    def test_rgb_to_ansi_fg(self):
        """Should generate correct ANSI true color foreground code."""
        from ascii_renderer.color import rgb_to_ansi_fg
        result = rgb_to_ansi_fg(255, 128, 0)
        assert result == "\x1b[38;2;255;128;0m"

    def test_rgb_to_ansi_bg(self):
        """Should generate correct ANSI true color background code."""
        from ascii_renderer.color import rgb_to_ansi_bg
        result = rgb_to_ansi_bg(0, 255, 128)
        assert result == "\x1b[48;2;0;255;128m"

    def test_colorize(self):
        """Should wrap character in color codes with reset."""
        from ascii_renderer.color import colorize, ANSI_RESET
        result = colorize("X", 255, 0, 0)
        assert result == "\x1b[38;2;255;0;0mX\x1b[0m"
        assert result.endswith(ANSI_RESET)

    def test_colorize_line_basic(self):
        """Should colorize a line of characters."""
        from ascii_renderer.color import colorize_line
        chars = ["A", "B", "C"]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = colorize_line(chars, colors)
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "\x1b[38;2;255;0;0m" in result
        assert "\x1b[0m" in result  # Reset at end

    def test_colorize_line_optimizes_same_color(self):
        """Should not emit redundant color codes for same color."""
        from ascii_renderer.color import colorize_line
        chars = ["A", "B", "C"]
        colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0)]  # All same
        result = colorize_line(chars, colors)
        # Should only have one color code (plus reset)
        assert result.count("\x1b[38;2;255;0;0m") == 1

    def test_colorize_line_empty(self):
        """Should handle empty input."""
        from ascii_renderer.color import colorize_line
        result = colorize_line([], [])
        assert result == ""


class TestColoredRendering:
    """Tests for colored ASCII rendering."""

    def test_render_with_color_flag(self):
        """Rendering with color=True should include ANSI codes."""
        from ascii_renderer.processor import load_image, render_to_ascii, RenderConfig

        # Create a simple colored image
        rgb = np.zeros((50, 100, 3), dtype=np.float32)
        rgb[:, :50, 0] = 1.0  # Left half red
        rgb[:, 50:, 2] = 1.0  # Right half blue
        luminance = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]

        config = RenderConfig(width=20, color=True)
        result = render_to_ascii(luminance, config, rgb=rgb)

        # Should contain ANSI escape codes
        assert "\x1b[38;2;" in result
        assert "\x1b[0m" in result

    def test_render_without_color_flag(self):
        """Rendering with color=False should not include ANSI codes."""
        image = np.random.rand(50, 100).astype(np.float32)
        config = RenderConfig(width=20, color=False)
        result = render_to_ascii(image, config)

        # Should not contain ANSI escape codes
        assert "\x1b[" not in result

    def test_sample_cell_color(self):
        """Should sample average color from a cell."""
        from ascii_renderer.processor import sample_cell_color

        # Create image with known color
        rgb = np.zeros((100, 100, 3), dtype=np.float32)
        rgb[:, :, 0] = 1.0  # All red

        r, g, b = sample_cell_color(rgb, 0, 0, 50, 50)
        assert r == 255
        assert g == 0
        assert b == 0
