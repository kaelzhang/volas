from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1280, 640
FRAME_MS = 95
OUTPUT = Path(__file__).with_name("volas_stale_tail_explainer.gif")

BG = (246, 248, 250)
INK = (18, 24, 32)
MUTED = (91, 103, 116)
BORDER = (205, 212, 220)
PANEL = (255, 255, 255)
RAW = (213, 226, 245)
WARMUP = (197, 203, 211)
EMPTY = (238, 241, 245)
GREEN = (102, 199, 128)
GREEN_DARK = (33, 128, 72)
STALE = (255, 147, 147)
STALE_DARK = (197, 47, 47)
YELLOW = (255, 215, 98)
CURSOR = (245, 132, 38)
FROZEN = (203, 212, 203)
GHOST = (238, 220, 220)


def load_font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates.extend(
            [
                "/System/Library/Fonts/SFNSMono.ttf",
                "DejaVuSansMono.ttf",
                "Menlo.ttc",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Microsoft/Arial Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Microsoft/Arial.ttf",
                "DejaVuSans.ttf",
            ]
        )

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(31, bold=True)
FONT_H2 = load_font(24, bold=True)
FONT_TEXT = load_font(20)
FONT_SMALL = load_font(17)
FONT_MONO = load_font(19, mono=True)
FONT_MONO_SMALL = load_font(16, mono=True)
FONT_MONO_BOLD = load_font(20, bold=True, mono=True)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
) -> None:
    tw, th = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=font, fill=fill)


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont = FONT_TEXT,
    fill: tuple[int, int, int] = INK,
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: tuple[int, int, int] = PANEL,
    outline: tuple[int, int, int] = BORDER,
    text_fill: tuple[int, int, int] = INK,
    font: ImageFont.ImageFont = FONT_TEXT,
) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=2)
    centered(draw, box, text, font, text_fill)


def cell_track(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    count: int,
    cell_w: int,
    cell_h: int,
    colors: list[tuple[int, int, int]],
    *,
    outline: tuple[int, int, int] = (116, 125, 136),
    labels: list[str | None] | None = None,
) -> None:
    for i in range(count):
        left = x + i * cell_w
        box = (left, y, left + cell_w - 3, y + cell_h)
        draw.rounded_rectangle(box, radius=3, fill=colors[i], outline=outline, width=1)
        if labels and labels[i]:
            centered(draw, box, labels[i], FONT_MONO_SMALL)


def cursor(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, text: str = "valid_rows") -> None:
    draw.line((x, y0, x, y1), fill=CURSOR, width=5)
    draw.polygon([(x, y0 - 8), (x - 7, y0 - 20), (x + 7, y0 - 20)], fill=CURSOR)
    label(draw, (x + 8, y0 + 2), text, font=FONT_MONO_SMALL, fill=CURSOR)


def base_frame(scene: str, code: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((34, 24), "volas: after append, recompute only the stale tail", font=FONT_TITLE, fill=INK)
    pill(draw, (34, 70, 820, 106), code, font=FONT_MONO, fill=(255, 255, 255), outline=(218, 226, 235))
    pill(draw, (1000, 26, 1238, 62), scene, font=FONT_TEXT, fill=(240, 245, 250), outline=(208, 216, 225))
    return img, draw


def meta_panel(draw: ImageDraw.ImageDraw, *, computed: int, show_state: bool, stale: bool = False) -> None:
    draw.rounded_rectangle((872, 112, 1238, 548), radius=12, fill=PANEL, outline=BORDER, width=2)
    label(draw, (898, 136), "Data model", font=FONT_H2)
    label(draw, (898, 176), "Column::F64(Arc<Vec<f64>>)", font=FONT_MONO_SMALL, fill=MUTED)
    label(draw, (898, 204), "+ validity mask for NA", font=FONT_MONO_SMALL, fill=MUTED)
    draw.line((898, 240, 1212, 240), fill=BORDER, width=1)
    label(draw, (898, 262), "ComputedMeta", font=FONT_H2)
    label(draw, (914, 302), 'directive: "ema:20"', font=FONT_MONO_SMALL, fill=INK)
    label(draw, (914, 330), "lookback: 19", font=FONT_MONO_SMALL, fill=INK)
    label(draw, (914, 358), "valid_rows: cursor", font=FONT_MONO_SMALL, fill=CURSOR)
    label(draw, (914, 386), "state: last EMA", font=FONT_MONO_SMALL, fill=GREEN_DARK if show_state else MUTED)
    draw.line((898, 424, 1212, 424), fill=BORDER, width=1)
    counter_fill = (255, 249, 227) if not stale else (255, 235, 235)
    pill(
        draw,
        (898, 456, 1212, 506),
        f"cells computed this read = {computed}",
        fill=counter_fill,
        outline=YELLOW if not stale else STALE_DARK,
        font=FONT_MONO_SMALL,
    )


def render_scene_1(frame: int, total: int) -> Image.Image:
    img, draw = base_frame("1 / original compute", 'df["ema:20"]')
    x, top_y, cell_w, cell_h = 58, 154, 23, 34
    p = ease(frame / max(1, total - 1))
    computed = round(30 * p)

    label(draw, (58, 128), "close", font=FONT_TEXT, fill=MUTED)
    cell_track(draw, x, top_y, 30, cell_w, cell_h, [RAW] * 30)
    label(draw, (58, 218), "ema:20 cache", font=FONT_TEXT, fill=MUTED)
    colors = []
    for i in range(30):
        if i < 19:
            colors.append(WARMUP)
        elif i < computed:
            colors.append(GREEN)
        else:
            colors.append(EMPTY)
    cell_track(draw, x, top_y + 88, 30, cell_w, cell_h, colors)

    draw.line((x, top_y + 132, x + 19 * cell_w, top_y + 132), fill=WARMUP, width=5)
    label(draw, (x + 152, top_y + 138), "lookback = 19", font=FONT_MONO_SMALL, fill=MUTED)
    cursor_x = x + computed * cell_w
    cursor(draw, cursor_x, top_y + 48, top_y + 140)

    label(draw, (58, 366), "O(n) - full pass, done ONCE", font=FONT_H2, fill=GREEN_DARK)
    if computed == 30:
        pill(draw, (58, 422, 320, 472), "state = last EMA", fill=(235, 250, 239), outline=GREEN, font=FONT_MONO)
        pill(draw, (344, 422, 612, 472), "valid_rows = height = 30", fill=(255, 250, 235), outline=CURSOR, font=FONT_MONO_SMALL)
    meta_panel(draw, computed=computed, show_state=computed == 30)
    return img


def render_scene_2(frame: int, total: int) -> Image.Image:
    img, draw = base_frame("2 / append bars", "df.append(bars)")
    x, top_y, cell_w, cell_h = 58, 154, 23, 34
    p = ease(frame / max(1, total - 1))
    slide = round((1.0 - p) * 92)

    label(draw, (58, 128), "close", font=FONT_TEXT, fill=MUTED)
    cell_track(draw, x, top_y, 30, cell_w, cell_h, [RAW] * 30)
    for i in range(3):
        left = x + (30 + i) * cell_w + slide
        draw.rounded_rectangle((left, top_y, left + cell_w - 3, top_y + cell_h), radius=3, fill=blend((255, 240, 201), RAW, p), outline=CURSOR, width=2)

    label(draw, (58, 218), "ema:20 cache", font=FONT_TEXT, fill=MUTED)
    colors = [WARMUP if i < 19 else GREEN for i in range(30)] + [STALE] * 3
    cell_track(draw, x, top_y + 88, 33, cell_w, cell_h, colors)
    cursor(draw, x + 30 * cell_w, top_y + 48, top_y + 140)

    height_x = x + 33 * cell_w
    draw.line((height_x, top_y + 48, height_x, top_y + 140), fill=STALE_DARK, width=3)
    label(draw, (height_x - 54, top_y + 146), "height = 33", font=FONT_MONO_SMALL, fill=STALE_DARK)

    pill(draw, (58, 366, 322, 416), "height: 30 -> 33", fill=(240, 245, 250), outline=BORDER, font=FONT_MONO)
    pill(draw, (344, 366, 606, 416), "valid_rows = 30", fill=(255, 249, 235), outline=CURSOR, font=FONT_MONO)
    pill(draw, (58, 454, 606, 512), "STALE  (valid_rows=30 < height=33)", fill=(255, 235, 235), outline=STALE_DARK, text_fill=STALE_DARK, font=FONT_MONO)
    meta_panel(draw, computed=0, show_state=True, stale=True)
    return img


def render_scene_3(frame: int, total: int) -> Image.Image:
    img, draw = base_frame("3 / read appended frame", 'df["ema:20"]  # refreshes only the stale tail')
    x, cell_w, cell_h = 58, 23, 34
    top_y = 148
    p = frame / max(1, total - 1)

    label(draw, (58, 120), "A) Naive recompute", font=FONT_H2, fill=STALE_DARK)
    flash = 0.5 + 0.5 * abs(((frame % 16) / 8.0) - 1.0)
    naive_colors = [blend(GHOST, YELLOW, flash) for _ in range(33)]
    cell_track(draw, x, top_y, 33, cell_w, cell_h, naive_colors)
    pill(draw, (864, 132, 1006, 178), "O(n)", fill=(255, 235, 235), outline=STALE_DARK, text_fill=STALE_DARK, font=FONT_H2)
    pill(draw, (1028, 132, 1238, 178), "cells computed = 33", fill=(255, 247, 218), outline=YELLOW, font=FONT_MONO_SMALL)

    label(draw, (58, 304), "B) volas - refresh_computed", font=FONT_H2, fill=GREEN_DARK)
    tail_progress = max(0.0, min(1.0, (p - 0.18) / 0.46))
    tail_done = min(3, int(tail_progress * 3.7))
    pulse_slot = min(2, tail_done) if tail_done < 3 else 2

    volas_colors = []
    for i in range(33):
        if i < 30:
            volas_colors.append(FROZEN)
        else:
            j = i - 30
            if j < tail_done:
                volas_colors.append(GREEN)
            elif j == pulse_slot and tail_done < 3:
                volas_colors.append(blend(STALE, GREEN, 0.35 + 0.35 * flash))
            else:
                volas_colors.append(STALE)
    cell_track(draw, x, 344, 33, cell_w, cell_h, volas_colors)
    cursor(draw, x + 30 * cell_w, 320, 398)

    draw.rounded_rectangle((58, 410, 734, 458), radius=8, fill=(239, 244, 239), outline=(182, 195, 182), width=2)
    centered(draw, (58, 410, 734, 458), "[0, valid_rows) reused - 0 work", FONT_MONO, MUTED)
    pill(draw, (752, 410, 836, 458), "tail", fill=(255, 235, 235), outline=STALE_DARK, text_fill=STALE_DARK, font=FONT_MONO)

    draw.rounded_rectangle((864, 250, 1238, 548), radius=12, fill=PANEL, outline=BORDER, width=2)
    label(draw, (894, 274), "resume path", font=FONT_H2)
    label(draw, (910, 318), "has_stale_computed", font=FONT_MONO, fill=STALE_DARK)
    label(draw, (910, 354), "execute_resume(state)", font=FONT_MONO, fill=GREEN_DARK)
    label(draw, (910, 390), "update_computed_tail", font=FONT_MONO, fill=GREEN_DARK)
    counter = tail_done if p < 0.75 else 3
    pill(draw, (894, 430, 1208, 480), f"cells computed this read = {counter}", fill=(235, 250, 239), outline=GREEN, font=FONT_MONO_SMALL)
    label(draw, (894, 506), "O(new bars) - here 3, not 33", font=FONT_H2, fill=GREEN_DARK)

    if p > 0.72:
        pill(draw, (310, 538, 650, 594), "33 cells vs 3 cells", fill=(255, 255, 255), outline=INK, font=FONT_H2)
        label(draw, (682, 552), "O(n) vs O(new bars + lookback)", font=FONT_H2, fill=INK)
    if p > 0.82:
        pill(draw, (58, 486, 358, 532), "valid_rows = 33", fill=(255, 249, 235), outline=CURSOR, font=FONT_MONO)
        pill(draw, (382, 486, 660, 532), "state refreshed", fill=(235, 250, 239), outline=GREEN, font=FONT_MONO)
    return img


def make_frames() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []

    for i in range(44):
        frames.append(render_scene_1(i, 44))
        durations.append(FRAME_MS)
    for _ in range(10):
        frames.append(render_scene_1(43, 44))
        durations.append(150)

    for i in range(38):
        frames.append(render_scene_2(i, 38))
        durations.append(FRAME_MS)
    for _ in range(12):
        frames.append(render_scene_2(37, 38))
        durations.append(155)

    for i in range(76):
        frames.append(render_scene_3(i, 76))
        durations.append(FRAME_MS)
    for _ in range(16):
        frames.append(render_scene_3(75, 76))
        durations.append(160)

    return frames, durations


def main() -> None:
    frames, durations = make_frames()
    paletted = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames]
    paletted[0].save(
        OUTPUT,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
