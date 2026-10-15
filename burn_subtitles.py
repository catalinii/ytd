#!/usr/bin/env python3
"""Burn word-level subtitles from config.yaml into a cut clip,
crop to 1:1 and follow the speaker horizontally.

Default target:
  video  /root/saved/data/wsoS7Dy19vQ_4024.7_4060.54.mp4
  which is cut from original video wsoS7Dy19vQ at offset 4024.7s
  (cut start == first word start, cut end == last word end).

  Subtitles are read from /root/saved/config.yaml (original-video
  timestamps) and shifted by -offset so 0.0 == cut start.

What it does:
  1. Groups word subtitles into short karaoke phrases (<= MAX_WORDS).
  2. Tracks the speaker's horizontal position with Haar face detection
     (+ optional YOLO person detection if ultralytics is installed),
     smooths the trajectory and crops a 1:1 (720x720) window that
     follows the speaker left/right.
  3. Renders the active phrase centred on the square frame with the
     currently spoken word highlighted in yellow, rest in white, thick
     black outline, big bold font for mobile legibility.
  4. Preserves the original audio (ffmpeg mux).

Use the venv as required:
  /root/saved/venv/bin/python burn_subtitles.py [options]

Example:
  /root/saved/venv/bin/python burn_subtitles.py \
    --video /root/saved/data/wsoS7Dy19vQ_4024.7_4060.54.mp4 \
    --video-id wsoS7Dy19vQ --offset 4024.7 \
    --output /root/saved/data/wsoS7Dy19vQ_4024.7_4060.54_square_captioned.mp4
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------- helpers

def parse_offset_from_filename(path, default=None):
    """Filenames look like <id>_<start>_<end>.mp4 ; return start as float."""
    base = os.path.basename(path)
    m = re.search(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.mp4$", base)
    if m:
        return float(m.group(1))
    return default


def load_words(config_path, video_id, offset, cut_end=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if video_id not in cfg:
        raise SystemExit(f"video_id {video_id!r} not in {config_path}. keys={list(cfg.keys())}")
    subs = cfg[video_id].get("subtitles", [])
    words = []
    for s in subs:
        try:
            start = float(s["start"]) - offset
            end = float(s["end"]) - offset
        except Exception:
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        # keep words overlapping the cut [0, cut_end] (cut_end=None => no upper bound)
        # NOTE: strict inequalities would keep zero-overlap words sitting
        # exactly on the boundary (e.g. "life." ending exactly at the cut
        # start); those have no audio inside the clip.
        if end <= 1e-6:
            continue
        if start < -0.5:
            continue
        if cut_end is not None and start >= cut_end - 1e-6:
            # word starts at/after the cut end: no audio inside the clip
            # (a word starting exactly at the end has zero overlap; eps
            # covers float dust only; cut_end is the probed duration)
            continue
        words.append({"start": start, "end": end, "text": text})
    words.sort(key=lambda w: w["start"])
    # clamp negative starts to 0
    for w in words:
        if w["start"] < 0:
            w["start"] = 0.0
    apply_cut_start_capital(subs, words)
    return words


def apply_cut_start_capital(all_subs, words):
    """Always capitalize the first letter of the cut's first word.

    all_subs: subtitles in ORIGINAL timestamps (only kept for signature
      compatibility; the decision no longer depends on sentence context).
    words: cut-relative word list (modified in place, also returned).
    Every other letter keeps config.yaml casing untouched. If the first
    word starts with punctuation/quotes, the first alphabetic character
    is capitalized instead.
    """
    if not words:
        return words
    first = words[0]["text"]
    chars = list(first)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            if ch.islower():
                chars[i] = ch.upper()
                words[0]["text"] = "".join(chars)
                print("[info] first word %r -> %r (capitalized)"
                      % (first, words[0]["text"]), flush=True)
            return words
    return words


def group_phrases(words, max_words=4, max_gap=0.6):
    """Group word list into short caption phrases.

    Break after max_words, after sentence punctuation, or on long pauses.
    """
    phrases = []
    cur = []
    for i, w in enumerate(words):
        cur.append(w)
        is_last = (i == len(words) - 1)
        nxt = words[i + 1] if not is_last else None
        gap = (nxt["start"] - w["end"]) if nxt else 0
        ends_sentence = w["text"][-1:] in ".?!,;:"
        if len(cur) >= max_words or (ends_sentence and len(cur) >= 2) or gap > max_gap or is_last:
            phrases.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "words": list(cur),
            })
            cur = []
    return phrases


def find_active(phrase_list, t):
    for p in phrase_list:
        if p["start"] <= t < p["end"]:
            return p
    return None


def active_word_index(phrase, t):
    idx = 0
    for i, w in enumerate(phrase["words"]):
        if w["start"] <= t:
            idx = i
        else:
            break
    # if t is past the end of the indexed word but before next word start,
    # keep highlighting the last started word (standard karaoke behaviour)
    return idx


# ------------------------------------------------------- speaker tracking

def detect_centers_haar(video_path, sample_every=5):
    """Return (centers_or_None per sampled frame, sample_indices, n_frames, w, h, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise SystemExit("failed to load Haar cascade")
    idxs = list(range(0, n, sample_every))
    raw = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            raw.append(None)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            raw.append(None)
        else:
            # largest face = speaker
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            raw.append(float(x + fw / 2.0))
    cap.release()
    return raw, idxs, n, w, h, float(fps)


def try_yolo_centers(video_path, sample_every=5):
    """Try ultralytics YOLOv8 person tracking. Returns list or None."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[track] ultralytics not available ({e}), using Haar", flush=True)
        return None
    try:
        import torch
        print(f"[track] trying YOLOv8n person detection (torch {torch.__version__})",
              flush=True)
        model = YOLO("yolov8n.pt")  # auto-downloads ~6MB on first use
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = list(range(0, n, sample_every))
        raw = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                raw.append(None)
                continue
            res = model.predict(frame, classes=[0], verbose=False, conf=0.4)
            best = None
            best_area = 0
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1 + x2) / 2.0
            raw.append(best)
        cap.release()
        n_det = sum(1 for v in raw if v is not None)
        print(f"[track] YOLO detections: {n_det}/{len(raw)}", flush=True)
        if n_det < len(raw) * 0.2:
            print("[track] too few YOLO detections, falling back to Haar",
                  flush=True)
            return None
        return raw
    except Exception as e:
        print(f"[track] YOLO failed ({e}), using Haar", flush=True)
        return None


def build_crop_trajectory(n_frames, frame_w, crop_s, raw, idxs, sample_every):
    """Interpolate + smooth raw centers -> per-frame crop x0 in [0, frame_w-crop_s]."""
    max_x0 = frame_w - crop_s
    center_default = frame_w / 2.0
    # forward/backward fill missing detections
    filled = list(raw)
    last = None
    for i, v in enumerate(filled):
        if v is None:
            filled[i] = last
        else:
            last = v
    if filled and filled[0] is None:
        # no detection at start: use first valid or frame center
        first_valid = next((v for v in filled if v is not None), center_default)
        filled = [first_valid if v is None else v for v in filled]
    if not filled:
        filled = [center_default]
    # moving average over sampled points (window 7 ~ 1s at every=5,30fps)
    win = 7
    sm = []
    for i in range(len(filled)):
        lo = max(0, i - win // 2)
        hi = min(len(filled), i + win // 2 + 1)
        sm.append(float(np.mean(filled[lo:hi])))
    # interpolate sampled -> every frame
    sampled_x = np.array(idxs, dtype=float)
    sampled_c = np.array(sm, dtype=float)
    all_frames = np.arange(n_frames, dtype=float)
    interp = np.interp(all_frames,
                       sampled_x,
                       sampled_c,
                       left=sampled_c[0], right=sampled_c[-1])
    # light exponential smoothing to kill jitter, then speed clamp
    alpha = 0.18
    out = [interp[0]]
    for v in interp[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    out = np.array(out)
    # clamp max horizontal speed (~6 px/frame @30fps = 180px/s)
    max_dx = 6.0
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > max_dx:
            out[i] = out[i - 1] + max_dx
        elif d < -max_dx:
            out[i] = out[i - 1] - max_dx
    x0 = np.clip(out - crop_s / 2.0, 0, max_x0).astype(int)
    return x0


# ------------------------------------------------------------- rendering

# Fonts ship with the repo (fonts/ next to this file); resolve relative to
# this file so it works wherever the repo is checked out
# (/root/saved, /usr/src/saved, ...), not just one absolute path.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FONT_PATH = os.path.join(_BASE_DIR, "fonts", "LibreBaskerville-Bold.ttf")
DEFAULT_FONT_REGULAR = os.path.join(_BASE_DIR, "fonts", "LibreBaskerville-Regular.ttf")

def load_fonts(size, emphasize_scale=1.3, font_path=None):
    """Load (base, big) fonts. Big is used for important words."""
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates += [
        DEFAULT_FONT_PATH,
        DEFAULT_FONT_REGULAR,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    base = None
    used = None
    for p in candidates:
        if p and os.path.exists(p):
            try:
                base = ImageFont.truetype(p, size)
                used = p
                break
            except Exception:
                continue
    if base is None:
        base = ImageFont.load_default()
        used = "<default>"
    big_size = int(round(size * emphasize_scale))
    try:
        big = ImageFont.truetype(used, big_size) if used != "<default>" else ImageFont.load_default()
    except Exception:
        big = base
    print(f"[font] base={used} {size}px, emphasized {big_size}px (x{emphasize_scale})",
          flush=True)
    return base, big


def _clean_token(text):
    return text.strip().strip(".,?!;:\"'“”‘’()[]").lower()


# Static default important-words set, proposed from config.yaml:
# - top content words of this cut (4024.7-4060.54): power, curses, barren,
#   generosity, Elisha, children, ...
# - top sermon themes of full wsoS7Dy19vQ video: god, jesus, room, life, ...
DEFAULT_IMPORTANT_WORDS = frozenset({
    "god", "god's", "jesus", "lord", "holy", "bible", "church", "pray",
    "faith", "redeemed", "room", "access", "house", "people", "life",
    "power", "curses", "curse", "barren", "inherited", "bearing", "born",
    "woman", "children", "elisha", "generosity", "hospitable", "listen",
    "opposite", "ambition", "door",
})


def is_important(text, idx_in_phrase, min_len=7, extra_set=None):
    """Heuristic importance: long content words, mid-phrase proper nouns,
    numbers, or an explicit --important-words list. Returns bool."""
    c = _clean_token(text)
    if not c:
        return False
    if extra_set and c in extra_set:
        return True
    if any(ch.isdigit() for ch in c):
        return True
    if len(c) >= min_len:
        return True
    # proper noun heuristic: Capitalized but not the first word of the phrase
    # (whisper capitalizes sentence starts, so mid-phrase caps ~ names/God)
    stripped = text.strip()
    if idx_in_phrase > 0 and stripped[:1].isupper():
        return True
    return False


def wrap_linesMixed(word_items, max_width, draw):
    """word_items: list of dicts {text, font, ...}. Greedy wrap.

    Returns ALL lines uncapped: dropping overflow lines here once silently
    deleted the caption's first line (often the highlighted word), so the
    caller (draw_caption) now shrinks fonts to fit and only ever drops
    lines while keeping the active word visible.
    """
    lines, cur, cur_w = [], [], 0
    # space width measured with base font (first item's font fallback)
    for it in word_items:
        ww = draw.textlength(it["text"], font=it["font"])
        # space width depends on current word font; use its font
        sw = draw.textlength(" ", font=it["font"])
        it["_w"] = ww
        it["_sw"] = sw
    for it in word_items:
        add = it["_w"] if not cur else it["_sw"] + it["_w"]
        # NOTE: for simplicity use the incoming word's space width;
        # visually negligible.
        if cur and cur_w + add > max_width:
            lines.append(cur)
            cur, cur_w = [], 0
            add = it["_w"]
        cur.append(it)
        cur_w += add
    if cur:
        lines.append(cur)
    return lines


MIN_BASE_FONT_SIZE = 40  # shrink-to-fit floor keeps captions mobile-readable


def _scale_font(font, factor, floor):
    """Return a size*factor copy of a TrueType font (same file).

    Returns the SAME object when scaling is impossible (bitmap fallback)
    or the floor is reached, so callers can detect it via identity.
    """
    try:
        path = font.path
        size = font.size
    except AttributeError:
        return font
    new_size = max(floor, int(size * factor))
    if new_size >= size:
        return font
    try:
        return ImageFont.truetype(path, new_size)
    except Exception:
        return font


def draw_caption(pil_img, phrase, t, font_base, font_big,
                 max_width_ratio=0.86, min_emphasize_len=7, extra_set=None,
                 enable_emphasize=True):
    """Draw active phrase lower-third; highlight active word yellow.

    Important words (long / proper nouns / --important-words) are rendered
    with font_big (larger). Active word is yellow regardless of size.
    Phrases that would wrap past 2 lines shrink both fonts to fit (down to
    MIN_BASE_FONT_SIZE); as a last resort the kept 2-line window always
    contains the active word, so it can never be silently dropped.
    """
    if phrase is None:
        return pil_img
    W, H = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    words = [w["text"] for w in phrase["words"]]
    ai = active_word_index(phrase, t)
    max_width = int(W * max_width_ratio)
    flags = []
    for i, w in enumerate(words):
        imp = (enable_emphasize and
               is_important(w, i, min_len=min_emphasize_len, extra_set=extra_set))
        flags.append((w, i == ai, imp))
    cur_base, cur_big = font_base, font_big
    lines = []
    for _ in range(12):  # bounded shrink-to-fit loop
        items = [{"text": w, "font": cur_big if imp else cur_base,
                  "is_active": act, "is_important": imp}
                 for (w, act, imp) in flags]
        lines = wrap_linesMixed(items, max_width, draw)
        if len(lines) <= 2:
            break
        nxt_base = _scale_font(cur_base, 0.94, MIN_BASE_FONT_SIZE)
        nxt_big = _scale_font(cur_big, 0.94, int(MIN_BASE_FONT_SIZE * 1.3))
        if nxt_base is cur_base and nxt_big is cur_big:
            break
        cur_base, cur_big = nxt_base, nxt_big
    if len(lines) > 2:
        ai_line = next((li for li, ln in enumerate(lines)
                        if any(it["is_active"] for it in ln)), 0)
        first_ln = max(0, min(ai_line, len(lines) - 2))
        lines = lines[first_ln:first_ln + 2]
    # per-line heights (accommodate mixed sizes), 8px line gap
    line_heights, line_ascents = [], []
    for line in lines:
        asc = max(f["font"].getmetrics()[0] for f in line)
        desc = max(f["font"].getmetrics()[1] for f in line)
        line_ascents.append(asc)
        line_heights.append(asc + desc + 8)
    block_h = sum(line_heights)
    # lower-third: block bottom ends 10% above screen bottom
    y = int(H * 0.90) - block_h
    YELLOW = (255, 235, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    for line, lh, max_asc in zip(lines, line_heights, line_ascents):
        total = sum(it["_w"] for it in line) + sum(it["_sw"] for it in line[1:])
        x = (W - total) / 2
        for j, it in enumerate(line):
            asc, _ = it["font"].getmetrics()
            # baseline-align mixed sizes within the line
            yy = y + (max_asc - asc)
            fill = YELLOW if it["is_active"] else WHITE
            # slightly thicker outline for the big emphasized words
            sw = 5 if it["is_important"] else 4
            draw.text((x, yy), it["text"], font=it["font"], fill=fill,
                      stroke_width=sw, stroke_fill=BLACK)
            x += it["_w"] + (it["_sw"] if j < len(line) - 1 else 0)
        y += lh
    return pil_img


# ------------------------------------------- header banner / IG canvas
# Instagram-ready canvas: the speaker-following square is kept intact and a
# solid header strip is stacked on top for a persistent message (hook/CTA).
#  - square / 1:1 : legacy 720x720, no header (default, unchanged behaviour)
#  - 4:5        : 720x900  (header 180px) - max feed surface, recommended
#                 for posts like "Free counseling for life problems"
#  - 9:16       : 720x1280 (header ~336px + footer ~224px) - Reels/Stories
# Captions are still drawn on the square video region first, then the
# captioned square is pasted below the header, so caption position never
# shifts into the banner.

ASPECT_PRESETS = {
    "square": (1, 1),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "4:6": (4, 6),
    "9:16": (9, 16),
}

DEFAULT_HEADER_BG = (17, 24, 39)   # dark navy - high contrast, calm
DEFAULT_HEADER_FG = (255, 255, 255)
DEFAULT_ACCENT_FG = (255, 235, 0)  # same yellow as active caption word


def parse_hex_color(value, default):
    """Accept '#rrggbb', 'rrggbb', or 'r,g,b'. Returns (r, g, b)."""
    if value is None:
        return default
    v = str(value).strip()
    if "," in v:
        try:
            parts = [int(p.strip()) for p in v.split(",")]
            if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                return tuple(parts)
        except ValueError:
            return default
        return default
    v = v.lstrip("#")
    if len(v) == 6:
        try:
            return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
        except ValueError:
            return default
    return default


def compute_canvas(crop_s, aspect="square", header_h=None, footer_h=None):
    """Return (out_w, out_h, header_px, footer_px) for a square video.

    crop_s: side of the speaker-following square (e.g. 720).
    aspect: one of square|1:1|4:5|9:16.
    header_h/footer_h: explicit overrides in square-crop pixels (None = preset).
    """
    key = str(aspect or "square").strip().lower().replace(" ", "")
    if key not in ASPECT_PRESETS:
        raise ValueError(f"unknown aspect {aspect!r}; "
                         f"expected one of {sorted(ASPECT_PRESETS)}")
    if key in ("square", "1:1"):
        hh = int(header_h or 0)
        fh = int(footer_h or 0)
        if hh < 0 or fh < 0:
            raise ValueError("header/footer heights must be >= 0")
        return (crop_s, crop_s + hh + fh, hh, fh)
    ar_w, ar_h = ASPECT_PRESETS[key]
    total_h = int(round(crop_s * ar_h / ar_w))
    extra = max(0, total_h - crop_s)
    if header_h is not None or footer_h is not None:
        hh = int(header_h or 0)
        fh = int(footer_h if footer_h is not None else (extra - hh))
        if hh < 0 or fh < 0 or hh + fh != extra:
            raise ValueError(
                f"header ({hh}) + footer ({fh}) must equal {extra} "
                f"for {key} at width {crop_s}")
        return (crop_s, crop_s + hh + fh, hh, fh)
    if key == "9:16":
        # Split the tall extra strip: roomy hook on top, clean footer below.
        hh = int(round(extra * 0.6))
        fh = extra - hh
        return (crop_s, total_h, hh, fh)
    # 4:5 etc: all extra space on top as the message banner.
    return (crop_s, total_h, extra, 0)


def _fit_header_font(text, bar_w, bar_h, font_path=None, start_size=64,
                     floor=24, pad_x=36, pad_y=18):
    """Pick the largest Bold font size so wrapped text fits bar_w x bar_h."""
    from PIL import Image as _I, ImageDraw as _D
    probe = _I.new("RGB", (bar_w, max(bar_h, 1)))
    draw = _D.Draw(probe)
    size = start_size
    while True:
        try:
            font = (ImageFont.truetype(font_path or DEFAULT_FONT_PATH, size)
                    if os.path.exists(font_path or DEFAULT_FONT_PATH)
                    else ImageFont.load_default())
        except Exception:
            font = ImageFont.load_default()
        max_w = bar_w - 2 * pad_x
        words = str(text).split()
        lines, cur = [], ""
        for w_ in words:
            trial = (cur + " " + w_).strip()
            if cur and draw.textlength(trial, font=font) > max_w:
                lines.append(cur)
                cur = w_
            else:
                cur = trial
        if cur:
            lines.append(cur)
        try:
            asc, desc = font.getmetrics()
            lh = asc + desc + 6
        except Exception:
            lh = size + 6
        if len(lines) * lh <= max(1, bar_h - 2 * pad_y) or size <= floor:
            return font, lines
        size = max(floor, int(size * 0.92))
        if size == floor:
            # one final pass at floor size
            continue


def render_text_bar(bar_w, bar_h, text, font_path=None, bg=DEFAULT_HEADER_BG,
                    fg=DEFAULT_HEADER_FG, start_size=64):
    """Render a solid centered multi-line text bar (PIL Image)."""
    bar = Image.new("RGB", (bar_w, max(bar_h, 1)), tuple(bg))
    if not text or bar_h <= 0:
        return bar
    font, lines = _fit_header_font(text, bar_w, bar_h, font_path,
                                   start_size=start_size)
    draw = ImageDraw.Draw(bar)
    try:
        asc, desc = font.getmetrics()
        lh = asc + desc + 6
    except Exception:
        lh = font.size + 6 if hasattr(font, "size") else 30
    y = (bar_h - len(lines) * lh) / 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        draw.text(((bar_w - lw) / 2, y), line, font=font,
                  fill=tuple(fg), stroke_width=2, stroke_fill=(0, 0, 0))
        y += lh
    return bar


# ------------------------------------------------- reusable entry point
# (imported by app.py so /generate can produce the captioned square cut
# in the same request; the CLI main() below is a thin wrapper around it).

def render_captioned_square(video_path, words, output_path, max_words=4,
                            font_size=58, font_path=None, emphasize_scale=1.3,
                            min_emphasize_len=7, extra_set=None,
                            enable_emphasize=True, sample_every=5,
                            use_yolo=False, header_text="", aspect="square",
                            header_height=None, footer_height=None,
                            footer_text="", out_width=None,
                            header_bg=None, header_fg=None):
    """Burn karaoke captions into a 1:1 speaker-following crop.

    video_path: cut clip to process (audio is muxed from here).
    words: list of {"start","end","text"} in CUT-relative seconds.
    output_path: where to write the captioned mp4.
    extra_set: important-words set; None selects DEFAULT_IMPORTANT_WORDS.
    header_text: persistent banner message drawn in a strip ABOVE the
      square video (e.g. "Free counseling for life problems - DM us").
      Empty = no banner text (strip still added when aspect != square).
    aspect: "square" (legacy 1:1), "4:5" (IG feed, recommended with a
      header), or "9:16" (Reels/Stories: header + footer fill).
    header_height/footer_height: explicit strip heights in crop pixels;
      default follows the aspect preset. header+footer must equal the
      preset extra height.
    footer_text: optional small line in the 9:16 bottom strip.
    out_width: upscale final canvas width (e.g. 1080 for Instagram).
      Height scales proportionally. None = keep crop width (720).
    header_bg/header_fg: banner colors as "#rrggbb" or "r,g,b".
    Returns output_path. Raises ValueError/SystemExit on failure.
    """
    if not words:
        raise ValueError("no subtitles in cut range")
    if extra_set is None:
        extra_set = DEFAULT_IMPORTANT_WORDS
        print(f"[font] default important-words ({len(extra_set)}): "
              f"{', '.join(sorted(extra_set))}", flush=True)
    phrases = group_phrases(words, max_words=max_words)
    print(f"[info] {len(phrases)} caption phrases (max {max_words} words each)",
          flush=True)

    raw_yolo = try_yolo_centers(video_path, sample_every) if use_yolo else None
    raw, idxs, n_frames, W, H, fps = detect_centers_haar(
        video_path, sample_every)
    if raw_yolo is not None and len(raw_yolo) == len(raw):
        # fuse: prefer YOLO person center, fall back to Haar face
        raw = [y if y is not None else h for y, h in zip(raw_yolo, raw)]
        print("[track] fused YOLO person + Haar face", flush=True)
    n_det = sum(1 for v in raw if v is not None)
    print(f"[track] Haar face detections: {n_det}/{len(raw)} over {n_frames} frames "
          f"({W}x{H} @ {fps:.1f}fps)", flush=True)

    crop_s = min(W, H)
    x0_traj = build_crop_trajectory(n_frames, W, crop_s, raw, idxs,
                                    sample_every)
    print(f"[track] square crop {crop_s}x{crop_s}, x0 range "
          f"[{int(x0_traj.min())},{int(x0_traj.max())}]", flush=True)

    bg = parse_hex_color(header_bg, DEFAULT_HEADER_BG)
    fg = parse_hex_color(header_fg, DEFAULT_HEADER_FG)
    out_w, out_h, header_px, footer_px = compute_canvas(
        crop_s, aspect=aspect, header_h=header_height,
        footer_h=footer_height)
    scale = 1.0
    if out_width:
        scale = float(out_width) / float(out_w)
        out_w = int(out_width)
        out_h = int(round(out_h * scale))
        header_px = int(round(header_px * scale))
        footer_px = int(round(footer_px * scale))
        crop_s_out = int(round(crop_s * scale))
        # libx264 yuv420p needs even dimensions; keep video region exact
        # and absorb odd pixels into the footer/header fill.
        if out_w % 2:
            out_w -= 1
        if out_h % 2:
            out_h -= 1
        if (header_px + crop_s_out + footer_px) != out_h:
            footer_px = max(0, out_h - header_px - crop_s_out)
    else:
        crop_s_out = crop_s
    print(f"[canvas] aspect={aspect} canvas={out_w}x{out_h} "
          f"header={header_px}px footer={footer_px}px "
          f"header_text={header_text!r}", flush=True)

    header_bar = (render_text_bar(out_w, header_px, header_text,
                                  font_path=font_path, bg=bg, fg=fg)
                  if header_px > 0 else None)
    footer_bar = (render_text_bar(out_w, footer_px, footer_text,
                                  font_path=font_path, bg=bg, fg=fg,
                                  start_size=44)
                  if footer_px > 0 else None)

    font_base, font_big = load_fonts(font_size,
                                     emphasize_scale=emphasize_scale,
                                     font_path=font_path)
    cap = cv2.VideoCapture(video_path)
    tmp_fd, tmp_silent = tempfile.mkstemp(suffix="_silent.mp4")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_silent, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise SystemExit("cannot open VideoWriter")
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = fi / fps
        x0 = int(x0_traj[fi]) if fi < len(x0_traj) else int(x0_traj[-1])
        crop = frame[0:crop_s, x0:x0 + crop_s]
        if crop.shape[1] != crop_s or crop.shape[0] != crop_s:
            crop = cv2.resize(crop, (crop_s, crop_s))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        phrase = find_active(phrases, t)
        if phrase is not None:
            # Captions anchor to the SQUARE video region (lower-third of
            # the square), never to the full canvas, so the banner stays clean.
            pil = draw_caption(pil, phrase, t, font_base, font_big,
                               min_emphasize_len=min_emphasize_len,
                               extra_set=extra_set,
                               enable_emphasize=enable_emphasize)
        if scale != 1.0:
            pil = pil.resize((crop_s_out, crop_s_out), Image.LANCZOS)
        if header_px == 0 and footer_px == 0:
            canvas = pil
        else:
            canvas = Image.new("RGB", (out_w, out_h), tuple(bg))
            if header_bar is not None:
                canvas.paste(header_bar, (0, 0))
            canvas.paste(pil, (0, header_px))
            if footer_bar is not None:
                canvas.paste(footer_bar, (0, header_px + crop_s_out))
        back = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        out.write(back)
        fi += 1
        if fi % 150 == 0:
            print(f"[render] {fi}/{n_frames} frames", flush=True)
    cap.release()
    out.release()
    print(f"[render] wrote {fi} frames -> {tmp_silent}", flush=True)

    cmd = ["ffmpeg", "-y", "-i", tmp_silent, "-i", video_path,
           "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "medium", "-c:a", "aac", "-shortest", output_path]
    print("[mux] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise SystemExit("ffmpeg mux failed")
    try:
        os.remove(tmp_silent)
    except OSError:
        pass
    print(f"[done] {output_path}", flush=True)
    return output_path


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="1:1 speaker-follow + karaoke captions")
    ap.add_argument("--video", default="/root/saved/data/wsoS7Dy19vQ_4024.7_4060.54.mp4")
    ap.add_argument("--video-id", default="wsoS7Dy19vQ")
    ap.add_argument("--offset", type=float, default=None,
                    help="original-video timestamp of cut start (default: parsed from filename)")
    ap.add_argument("--config", default="/root/saved/config.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-words", type=int, default=4)
    ap.add_argument("--font-size", type=int, default=58,
                    help="base caption font px for 720px square output (Libre Baskerville Bold)")
    ap.add_argument("--font", default=None,
                    help="path to .ttf to use (default: Libre Baskerville Bold)")
    ap.add_argument("--emphasize-scale", type=float, default=1.3,
                    help="size multiplier for important words (0 to disable via --no-emphasize)")
    ap.add_argument("--min-emphasize-len", type=int, default=7,
                    help="words with >=N letters are emphasized (larger)")
    ap.add_argument("--important-words", default=None,
                    help="comma-separated words to emphasize INSTEAD of the built-in "
                         "DEFAULT_IMPORTANT_WORDS set; e.g. 'god,faith,covenant'")
    ap.add_argument("--no-emphasize", action="store_true",
                    help="disable larger-font emphasis; all words same size")
    ap.add_argument("--sample-every", type=int, default=5,
                    help="face-detect every N frames")
    ap.add_argument("--no-yolo", action="store_true",
                    help="skip YOLO attempt even if ultralytics installed")
    ap.add_argument("--header-text", default="",
                    help="persistent banner above the square video, e.g. "
                         "'Free counseling for life problems - DM us'")
    ap.add_argument("--aspect", default="square",
                    help="final canvas: square (720x720 legacy), 4:5 "
                         "(720x900 IG feed, recommended with --header-text), "
                         "9:16 (720x1280 Reels/Stories)")
    ap.add_argument("--header-height", type=int, default=None,
                    help="override header strip height in crop pixels")
    ap.add_argument("--footer-height", type=int, default=None,
                    help="override footer strip height in crop pixels")
    ap.add_argument("--footer-text", default="",
                    help="optional line in the 9:16 bottom strip")
    ap.add_argument("--out-width", type=int, default=None,
                    help="upscale final canvas width, e.g. 1080 for Instagram "
                         "(height scales proportionally)")
    ap.add_argument("--header-bg", default=None,
                    help="banner background '#rrggbb' or 'r,g,b'")
    ap.add_argument("--header-fg", default=None,
                    help="banner text color '#rrggbb' or 'r,g,b'")
    args = ap.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise SystemExit(f"video not found: {video_path}")
    offset = args.offset
    if offset is None:
        offset = parse_offset_from_filename(video_path)
        if offset is None:
            raise SystemExit("cannot determine --offset; pass it explicitly")
    print(f"[info] video={video_path}\n[info] video_id={args.video_id} offset={offset}",
          flush=True)

    # probe duration so we only keep subtitles overlapping the cut
    _probe = cv2.VideoCapture(video_path)
    _n = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _fps = float(_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    _probe.release()
    cut_duration = _n / _fps if _fps > 0 else None
    print(f"[info] cut duration ~{cut_duration:.2f}s ({_n} frames @ {_fps:.1f}fps)",
          flush=True)

    words = load_words(args.config, args.video_id, offset, cut_end=cut_duration)
    print(f"[info] {len(words)} words in cut range", flush=True)
    if not words:
        raise SystemExit("no subtitles in cut range - check offset/video-id")
    print(f"[info] first: {words[0]}", flush=True)
    print(f"[info] last:  {words[-1]}", flush=True)
    if args.important_words:
        extra_set = ({w.strip().lower() for w in args.important_words.split(",")
                      if w.strip()} or None)
        print(f"[font] custom --important-words: {sorted(extra_set or [])}", flush=True)
    else:
        extra_set = None  # render_captioned_square() selects DEFAULT_IMPORTANT_WORDS
    enable_emphasize = (not args.no_emphasize) and args.emphasize_scale > 1.0

    if args.output is None:
        root, _ = os.path.splitext(video_path)
        args.output = root + "_square_captioned.mp4"
    render_captioned_square(
        video_path, words, args.output, max_words=args.max_words,
        font_size=args.font_size, font_path=args.font,
        emphasize_scale=args.emphasize_scale,
        min_emphasize_len=args.min_emphasize_len, extra_set=extra_set,
        enable_emphasize=enable_emphasize,
        sample_every=args.sample_every, use_yolo=not args.no_yolo,
        header_text=args.header_text, aspect=args.aspect,
        header_height=args.header_height, footer_height=args.footer_height,
        footer_text=args.footer_text, out_width=args.out_width,
        header_bg=args.header_bg, header_fg=args.header_fg)


if __name__ == "__main__":
    main()
