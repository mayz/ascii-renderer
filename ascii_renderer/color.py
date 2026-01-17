"""
ANSI color code utilities for colored ASCII output.

Supports 24-bit true color (works in modern terminals like ghostty, iTerm2, kitty).
"""

# ANSI escape code to reset all formatting
ANSI_RESET = "\x1b[0m"


def rgb_to_ansi_fg(r: int, g: int, b: int) -> str:
    """
    Return ANSI escape code for 24-bit true color foreground.

    Args:
        r, g, b: Color components (0-255).

    Returns:
        ANSI escape sequence string.
    """
    return f"\x1b[38;2;{r};{g};{b}m"


def rgb_to_ansi_bg(r: int, g: int, b: int) -> str:
    """
    Return ANSI escape code for 24-bit true color background.

    Args:
        r, g, b: Color components (0-255).

    Returns:
        ANSI escape sequence string.
    """
    return f"\x1b[48;2;{r};{g};{b}m"


def colorize(char: str, r: int, g: int, b: int) -> str:
    """
    Wrap a character in ANSI color codes.

    Args:
        char: The character to colorize.
        r, g, b: Foreground color components (0-255).

    Returns:
        Character wrapped in ANSI color codes with reset at end.
    """
    return f"{rgb_to_ansi_fg(r, g, b)}{char}{ANSI_RESET}"


def colorize_line(chars: list[str], colors: list[tuple[int, int, int]]) -> str:
    """
    Colorize a line of characters, optimizing by only emitting color codes when color changes.

    Args:
        chars: List of characters.
        colors: List of (r, g, b) tuples, same length as chars.

    Returns:
        String with ANSI color codes.
    """
    if not chars:
        return ""

    parts = []
    last_color = None

    for char, color in zip(chars, colors):
        if color != last_color:
            parts.append(rgb_to_ansi_fg(*color))
            last_color = color
        parts.append(char)

    parts.append(ANSI_RESET)
    return "".join(parts)
