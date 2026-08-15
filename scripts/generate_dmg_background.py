#!/usr/bin/env python3
"""Generate the branded macOS DMG background at 1x and 2x."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "resource" / "assets" / "dmg-background.png"
OUTPUT_2X = ROOT / "resource" / "assets" / "dmg-background@2x.png"

WIDTH = 720
HEIGHT = 440

COLORS = {
    "background": "#F4F7FC",
    "surface": "#FFFFFF",
    "surface_border": "#DFE6F1",
    "divider": "#E5EAF2",
    "title": "#171C2B",
    "body": "#667085",
    "muted": "#8B96A9",
    "accent": "#4676EE",
    "accent_soft": "#EAF0FF",
    "accent_border": "#BFCFFF",
}


def _font(size: int, *, bold: bool = False, scale: int = 1) -> ImageFont.FreeTypeFont:
    path = "/System/Library/Fonts/SFNS.ttf"
    try:
        return ImageFont.truetype(path, size * scale, index=1 if bold else 0)
    except OSError:
        fallback = "/System/Library/Fonts/Helvetica.ttc"
        return ImageFont.truetype(fallback, size * scale, index=1 if bold else 0)


def _cjk_font(size: int, *, scale: int = 1) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        size * scale,
    )


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    *,
    scale: int,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((center_x * scale - width / 2, top * scale), text, font=font, fill=fill)


def _step_badge(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top: int,
    number: str,
    label: str,
    *,
    scale: int,
) -> None:
    badge_box = (
        (center_x - 44) * scale,
        top * scale,
        (center_x - 20) * scale,
        (top + 24) * scale,
    )
    draw.rounded_rectangle(
        badge_box,
        radius=7 * scale,
        fill=COLORS["accent_soft"],
        outline=COLORS["accent_border"],
        width=1 * scale,
    )
    number_font = _font(11, bold=True, scale=scale)
    number_bounds = draw.textbbox((0, 0), number, font=number_font)
    number_width = number_bounds[2] - number_bounds[0]
    draw.text(
        (
            (center_x - 32) * scale - number_width / 2,
            (top + 4) * scale,
        ),
        number,
        font=number_font,
        fill=COLORS["accent"],
    )
    draw.text(
        ((center_x - 11) * scale, (top + 3) * scale),
        label,
        font=_cjk_font(12, scale=scale),
        fill=COLORS["body"],
    )


def _draw_arrow(draw: ImageDraw.ImageDraw, *, scale: int) -> None:
    center_y = 245 * scale
    start_x = 321 * scale
    end_x = 399 * scale
    draw.line(
        (start_x, center_y, end_x, center_y),
        fill=COLORS["accent_border"],
        width=2 * scale,
    )
    draw.rounded_rectangle(
        (343 * scale, 228 * scale, 377 * scale, 262 * scale),
        radius=11 * scale,
        fill=COLORS["accent"],
    )
    draw.line(
        (354 * scale, 245 * scale, 367 * scale, 245 * scale),
        fill="#FFFFFF",
        width=2 * scale,
    )
    draw.line(
        (363 * scale, 240 * scale, 368 * scale, 245 * scale, 363 * scale, 250 * scale),
        fill="#FFFFFF",
        width=2 * scale,
        joint="curve",
    )


def render(scale: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), COLORS["background"])
    draw = ImageDraw.Draw(image)

    _centered_text(
        draw,
        WIDTH // 2,
        35,
        "Install SubForge",
        _font(26, bold=True, scale=scale),
        COLORS["title"],
        scale=scale,
    )
    _centered_text(
        draw,
        WIDTH // 2,
        74,
        "将左侧应用拖到右侧 Applications 文件夹",
        _cjk_font(13, scale=scale),
        COLORS["body"],
        scale=scale,
    )

    draw.line(
        (40 * scale, 111 * scale, 680 * scale, 111 * scale),
        fill=COLORS["divider"],
        width=1 * scale,
    )

    for left, right in ((94, 286), (434, 626)):
        draw.rounded_rectangle(
            (left * scale, 139 * scale, right * scale, 349 * scale),
            radius=20 * scale,
            fill=COLORS["surface"],
            outline=COLORS["surface_border"],
            width=1 * scale,
        )

    _step_badge(draw, 190, 157, "1", "应用", scale=scale)
    _step_badge(draw, 530, 157, "2", "安装位置", scale=scale)
    _draw_arrow(draw, scale=scale)

    _centered_text(
        draw,
        WIDTH // 2,
        377,
        "拖放完成后，即可从“应用程序”启动 SubForge",
        _cjk_font(12, scale=scale),
        COLORS["muted"],
        scale=scale,
    )
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    render(1).save(OUTPUT, optimize=True)
    render(2).save(OUTPUT_2X, optimize=True)
    print(f"Generated {OUTPUT.relative_to(ROOT)}")
    print(f"Generated {OUTPUT_2X.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
