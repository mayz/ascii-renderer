"""
Command-line interface for ASCII renderer.
"""

import argparse
import sys

from .processor import RenderConfig, render_file_to_ascii


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ascii-render",
        description="Convert images to high-quality ASCII art using shape-based matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ascii-render image.png
  ascii-render image.jpg --width 120 --global-contrast 2.5
  ascii-render photo.png --edge-contrast 3.0 --output art.txt
  ascii-render dark-image.png --invert
  ascii-render colorful.png --color
        """,
    )

    parser.add_argument(
        "image",
        help="Path to input image file",
    )

    parser.add_argument(
        "-w", "--width",
        type=int,
        default=80,
        help="Output width in characters (default: 80)",
    )

    parser.add_argument(
        "-g", "--global-contrast",
        type=float,
        default=1.0,
        metavar="N",
        help="Global contrast enhancement exponent (default: 1.0, try 2-4 for sharper)",
    )

    parser.add_argument(
        "-e", "--edge-contrast",
        type=float,
        default=1.0,
        metavar="N",
        help="Directional edge contrast exponent (default: 1.0, try 2-4 for sharper edges)",
    )

    parser.add_argument(
        "-q", "--sample-quality",
        type=int,
        default=16,
        metavar="N",
        help="Samples per circle for quality (default: 16, higher = slower but better)",
    )

    parser.add_argument(
        "-a", "--aspect",
        type=float,
        default=0.5,
        metavar="RATIO",
        help="Character aspect ratio width/height (default: 0.5 for typical monospace)",
    )

    parser.add_argument(
        "-c", "--charset",
        type=str,
        default=None,
        help="Custom character set to use (default: printable ASCII)",
    )

    parser.add_argument(
        "-i", "--invert",
        action="store_true",
        help="Invert luminance (use for dark terminal backgrounds)",
    )

    parser.add_argument(
        "-C", "--color",
        action="store_true",
        help="Enable colored output (24-bit ANSI true color)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output file path (default: stdout)",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parsed = parse_args(args)

    config = RenderConfig(
        width=parsed.width,
        cell_aspect=parsed.aspect,
        global_contrast=parsed.global_contrast,
        edge_contrast=parsed.edge_contrast,
        sample_quality=parsed.sample_quality,
        charset=parsed.charset,
        invert=parsed.invert,
        color=parsed.color,
    )

    try:
        result = render_file_to_ascii(parsed.image, config)
    except FileNotFoundError:
        print(f"Error: Image file not found: {parsed.image}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        return 1

    if parsed.output:
        with open(parsed.output, "w") as f:
            f.write(result)
        print(f"Written to {parsed.output}")
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
