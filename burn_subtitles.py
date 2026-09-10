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
        if end < 0:
            continue
        if start < -0.5:
            continue
        if cut_end is not None and start > cut_end + 0.04:
            # one-frame tolerance only; the cut ends exactly at the last
            # word end, the next word starts after the cut
            continue
        words.append({"start": start, "end": end, "text": text})
    words.sort(key=lambda w: w["start"])
    # clamp negative starts to 0
    for w in words:
        if w["start"] < 0:
            w["start"] = 0.0
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

DEFAULT_FONT_PATH = "/root/saved/fonts/LibreBaskerville-Bold.ttf"

def load_fonts(size, emphasize_scale=1.3, font_path=None):
    """Load (base, big) fonts. Big is used for important words."""
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates += [
        DEFAULT_FONT_PATH,
        "/root/saved/fonts/LibreBaskerville-Regular.ttf",
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
    """word_items: list of dicts {text, font, ...}. Greedy wrap, cap 2 lines."""
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
    if len(lines) > 2:
        lines = lines[-2:]
    return lines


def draw_caption(pil_img, phrase, t, font_base, font_big,
                 max_width_ratio=0.86, min_emphasize_len=7, extra_set=None,
                 enable_emphasize=True):
    """Draw active phrase lower-third; highlight active word yellow.

    Important words (long / proper nouns / --important-words) are rendered
    with font_big (larger). Active word is yellow regardless of size.
    """
    if phrase is None:
        return pil_img
    W, H = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    words = [w["text"] for w in phrase["words"]]
    ai = active_word_index(phrase, t)
    max_width = int(W * max_width_ratio)
    items = []
    for i, w in enumerate(words):
        imp = (enable_emphasize and
               is_important(w, i, min_len=min_emphasize_len, extra_set=extra_set))
        items.append({"text": w, "font": font_big if imp else font_base,
                      "is_active": i == ai, "is_important": imp})
    lines = wrap_linesMixed(items, max_width, draw)
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


# ------------------------------------------------- reusable entry point
# (imported by app.py so /generate can produce the captioned square cut
# in the same request; the CLI main() below is a thin wrapper around it).

def render_captioned_square(video_path, words, output_path, max_words=4,
                            font_size=58, font_path=None, emphasize_scale=1.3,
                            min_emphasize_len=7, extra_set=None,
                            enable_emphasize=True, sample_every=5,
                            use_yolo=False):
    """Burn karaoke captions into a 1:1 speaker-following crop.

    video_path: cut clip to process (audio is muxed from here).
    words: list of {"start","end","text"} in CUT-relative seconds.
    output_path: where to write the captioned square mp4.
    extra_set: important-words set; None selects DEFAULT_IMPORTANT_WORDS.
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

    font_base, font_big = load_fonts(font_size,
                                     emphasize_scale=emphasize_scale,
                                     font_path=font_path)
    cap = cv2.VideoCapture(video_path)
    tmp_fd, tmp_silent = tempfile.mkstemp(suffix="_silent.mp4")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_silent, fourcc, fps, (crop_s, crop_s))
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
            pil = draw_caption(pil, phrase, t, font_base, font_big,
                               min_emphasize_len=min_emphasize_len,
                               extra_set=extra_set,
                               enable_emphasize=enable_emphasize)
        back = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
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
        sample_every=args.sample_every, use_yolo=not args.no_yolo)


if __name__ == "__main__":
    main()
