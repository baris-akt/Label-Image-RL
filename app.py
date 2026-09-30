#!/usr/bin/env python3
"""YOLO label web app. Combined viewer + crop search + change tracking.

    python app.py
    python3 app.py

Browse folders in the UI (File > Browse Images / Browse Annotations),
or paste a folder path and press Set.
"""
import os
import io
import sys
import json
import threading
import subprocess
import colorsys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import imagehash

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRACK_CACHE_DIR = os.path.join(DATA_DIR, "tracking_cache")
LAST_FOLDERS_FILE = os.path.join(DATA_DIR, "last_folders.txt")

IMAGES_DIR = ""
LABELS_DIR = ""

DEFAULT_CLASS_NAMES = ["bump-dent", "big-defect", "malzeme", "klinc"]
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)
HUE_BINS = 30
SAT_LEVELS = (1.0, 0.9, 0.8, 0.7)
VAL_LEVELS = (1.0, 0.80, 0.60, 0.40)
# class order walks hues in step-8 groups: 0,8,16... / 1,9,17... / ... / 7,15,23...
HUE_ORDER = [i for r in range(8) for i in range(r, HUE_BINS, 8)]


def class_sv(cls_idx):
    i = int(cls_idx) % 4
    return SAT_LEVELS[i], VAL_LEVELS[i]


def class_hex(cls_idx, sat=None, val=None):
    s0, v0 = class_sv(cls_idx)
    if sat is None:
        sat = s0
    if val is None:
        val = v0
    h = HUE_ORDER[int(cls_idx) % HUE_BINS] / float(HUE_BINS)
    r, g, b = colorsys.hsv_to_rgb(h, float(sat), float(val))
    return "#%02x%02x%02x" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PLACE_W, PLACE_H = 50, 50
BOX_THICKNESS, LABEL_SIZE = 2, 10
UNDO_MAX = 10
CLUSTER_RADIUS = 5.0
PHASH_SIZE = 16
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
FEATURE_DIM = PHASH_BITS + 256
CLUSTER_WORKERS = 8
CROP_ZOOM = 3.0
CROP_SMALL_FRAC = 0.07
CROP_SMALL_VIEW = 0.25
CROP_THUMB = 180
TRACK_YELLOW = "#FBEE00"
TRACK_BLUE = "#004482"

FILTER_PARAMS = {
    "gamma": {"value": 2.0},
    "clahe": {"clip": 2.0, "tile": 64},
    "unsharp": {"amount": 1.5, "sigma": 5.0},
    "emboss": {"strength": 1.0},
}

all_image_list = []
image_list = []
current_index = -1
class_filter_id = None
sort_mode = "name"
size_filter_on = False
size_filter_lo = 0.0
size_filter_hi = 1.0
path_box_size = {}
path_size_pct = {}
size_ordered_paths = []
box_size_cache = {}
box_size_cache_ready = False
path_wh = {}
crop_gen = 0
dataset_clustered = False
cluster_groups = []
cluster_ordered_paths = []
clustering = False
cluster_msg = ""
undo_stack = []
tracking_enabled = False
track_status = {}
crop_refs = []


def class_name(cls_idx):
    i = int(cls_idx)
    if 0 <= i < len(CLASS_NAMES) and str(CLASS_NAMES[i]).strip():
        return str(CLASS_NAMES[i]).strip()
    return "Unnamed-Cls%d" % i


def save_last_folders():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_FOLDERS_FILE, "w", encoding="utf-8") as f:
        f.write("%s\n%s\n" % (IMAGES_DIR, LABELS_DIR))


def load_last_folders():
    global IMAGES_DIR, LABELS_DIR
    if not os.path.isfile(LAST_FOLDERS_FILE):
        return
    with open(LAST_FOLDERS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n\r") for ln in f.readlines()]
    if len(lines) >= 1 and lines[0] and os.path.isdir(lines[0]):
        IMAGES_DIR = lines[0]
    if len(lines) >= 2 and lines[1] and os.path.isdir(lines[1]):
        LABELS_DIR = lines[1]


def classes_txt_path(folder=None):
    return os.path.join(folder if folder is not None else LABELS_DIR, "classes.txt")


def save_classes_txt():
    path = classes_txt_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write("%s\n" % name)


def load_classes_from_dir(folder):
    global CLASS_NAMES
    path = classes_txt_path(folder)
    if not os.path.isfile(path):
        return
    names = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            names.append(line.rstrip("\n\r"))
    if names:
        CLASS_NAMES = names


def list_images(folder):
    names = []
    if not os.path.isdir(folder):
        return names
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in IMG_EXTS:
                names.append(e.name)
    names.sort()
    return [os.path.join(folder, n) for n in names]


def label_path(img_path):
    return os.path.join(LABELS_DIR, Path(img_path).stem + ".txt")


def load_boxes(img_path):
    boxes = []
    p = label_path(img_path)
    if not os.path.isfile(p):
        return boxes
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append({
                "cls": max(0, int(float(parts[0]))),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return boxes


def write_boxes(path, boxes):
    global crop_refs
    p = label_path(path)
    with open(p, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (b["cls"], b["cx"], b["cy"], b["w"], b["h"]))
    crop_refs = []
    if box_size_cache_ready:
        cache_boxes_for_path(path, boxes)
        rebuild_size_order()


def copy_boxes(boxes):
    return [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"]} for b in boxes]


def image_has_class(path, cls_id):
    for b in load_boxes(path):
        if int(b["cls"]) == int(cls_id):
            return True
    return False


def image_wh(path):
    wh = path_wh.get(path)
    if wh is not None:
        return wh
    try:
        with Image.open(path) as im:
            wh = im.size
    except Exception:
        wh = (1, 1)
    path_wh[path] = wh
    return wh


def box_pixel_area(path, b):
    iw, ih = image_wh(path)
    return float(b["w"]) * float(b["h"]) * float(iw) * float(ih)


def cache_boxes_for_path(path, boxes=None):
    if boxes is None:
        boxes = load_boxes(path)
    entries = []
    mx = 0.0
    for b in boxes:
        area = box_pixel_area(path, b)
        e = {"cls": b["cls"], "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"], "area": area}
        entries.append(e)
        if area > mx:
            mx = area
    box_size_cache[path] = entries
    path_box_size[path] = mx


def rebuild_size_order():
    global size_ordered_paths, path_size_pct
    size_ordered_paths = sorted(all_image_list, key=lambda p: path_box_size.get(p, 0.0))
    n = len(size_ordered_paths)
    path_size_pct = {}
    if n == 0:
        return
    if n == 1:
        path_size_pct[size_ordered_paths[0]] = 1.0 if path_box_size.get(size_ordered_paths[0], 0.0) > 0 else 0.0
        return
    for i, p in enumerate(size_ordered_paths):
        path_size_pct[p] = i / float(n - 1)


def build_size_ranks():
    global box_size_cache_ready
    for p in all_image_list:
        cache_boxes_for_path(p)
    rebuild_size_order()
    box_size_cache_ready = True


def base_image_list():
    if sort_mode in ("size_asc", "size_desc"):
        if not box_size_cache_ready:
            build_size_ranks()
        if sort_mode == "size_desc":
            return list(reversed(size_ordered_paths))
        return list(size_ordered_paths)
    if sort_mode == "cluster" and dataset_clustered:
        return list(cluster_ordered_paths)
    return list(all_image_list)


def apply_class_filter(keep_path=None):
    global image_list, current_index, crop_refs
    crop_refs = []
    base = base_image_list()
    if class_filter_id is None:
        image_list = base
    else:
        image_list = [p for p in base if image_has_class(p, class_filter_id)]
    if size_filter_on and path_size_pct:
        lo = min(size_filter_lo, size_filter_hi)
        hi = max(size_filter_lo, size_filter_hi)
        image_list = [p for p in image_list if lo <= path_size_pct.get(p, 0.0) <= hi]
        if sort_mode in ("size_asc", "size_desc"):
            image_list.sort(key=lambda p: path_box_size.get(p, 0.0), reverse=(sort_mode == "size_desc"))
    if keep_path in image_list:
        current_index = image_list.index(keep_path)
    elif image_list:
        current_index = min(max(0, current_index), len(image_list) - 1)
    else:
        current_index = -1


def current_path():
    if image_list and 0 <= current_index < len(image_list):
        return image_list[current_index]
    return None


def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
                img = rgb[:, :, ::-1].copy()
        except Exception:
            img = None
    if img is None:
        return np.zeros((480, 640, 3), np.uint8)
    return img


def apply_filter(bgr, name):
    if not name or name not in FILTER_PARAMS:
        return bgr
    p = FILTER_PARAMS[name]
    if name == "gamma":
        g = max(0.05, float(p["value"]))
        inv = 1.0 / g
        table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(bgr, table)
    if name == "clahe":
        h, w = bgr.shape[:2]
        tile = max(2, int(p["tile"]))
        gw = max(1, w // tile)
        gh = max(1, h // tile)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(p["clip"]), tileGridSize=(gw, gh))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if name == "unsharp":
        blur = cv2.GaussianBlur(bgr, (0, 0), max(0.3, float(p["sigma"])))
        out = cv2.addWeighted(bgr, 1.0 + float(p["amount"]), blur, -float(p["amount"]), 0)
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "emboss":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        relief = np.clip((dx * 0.7071 + dy * 0.7071) * float(p["strength"]) + 128.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR)
    return bgr


def encode_jpg(bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return io.BytesIO(buf.tobytes()) if ok else io.BytesIO()


def extract_phash_feature(path):
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
            ph = imagehash.phash(img, hash_size=PHASH_SIZE)
            ph_bits = np.asarray(ph.hash, dtype=np.float32).flatten()
            gray = img.resize((64, 64)).convert("L")
            hist = np.asarray(gray.histogram(), dtype=np.float32)
        hist /= (hist.sum() + 1e-6)
        return np.concatenate([ph_bits, hist]).astype(np.float32)
    except Exception:
        return None


def track_key(path):
    return os.path.normpath(path)


def safe_token(s):
    out = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in str(s))
    return out.strip().strip(".") or "root"


def tracking_json_path():
    folder = Path(IMAGES_DIR)
    return os.path.join(
        TRACK_CACHE_DIR,
        "dataset-tracking-info-%s-%s.json" % (safe_token(folder.parent.name), safe_token(folder.name)),
    )


def save_tracking_json():
    if not tracking_enabled:
        return
    os.makedirs(TRACK_CACHE_DIR, exist_ok=True)
    payload = {
        "images_dir": IMAGES_DIR,
        "items": [{"path": p, "checked": bool(track_status.get(track_key(p), False))} for p in all_image_list],
    }
    with open(tracking_json_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_or_create_tracking():
    global track_status
    os.makedirs(TRACK_CACHE_DIR, exist_ok=True)
    jpath = tracking_json_path()
    if os.path.isfile(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        old = {}
        for it in data.get("items", []):
            if it.get("path"):
                old[track_key(it["path"])] = bool(it.get("checked", False))
        track_status = {}
        n_new = 0
        for p in all_image_list:
            k = track_key(p)
            if k in old:
                track_status[k] = old[k]
            else:
                track_status[k] = False
                n_new += 1
        n_checked = sum(1 for v in track_status.values() if v)
        save_tracking_json()
        return {
            "found": True,
            "images": len(track_status),
            "checked": n_checked,
            "unchecked": len(track_status) - n_checked,
            "new": n_new,
            "message": (
                "Found a matching tracking file and loaded saved progress.\n"
                "Images: %d\nChecked: %d\nUnchecked: %d\nNew images: %s"
                % (len(track_status), n_checked, len(track_status) - n_checked,
                   ("%d (marked unchecked)" % n_new) if n_new else "0")
            ),
        }
    track_status = {track_key(p): False for p in all_image_list}
    save_tracking_json()
    return {
        "found": False,
        "images": len(track_status),
        "checked": 0,
        "unchecked": len(track_status),
        "new": 0,
        "message": (
            "No tracking file was found for this dataset.\n"
            "A new tracking file was created with all images unmarked.\n\n"
            "Images: %d\nChecked: 0\nUnchecked: %d" % (len(track_status), len(track_status))
        ),
    }


def mark_checked(path):
    if not tracking_enabled or not path:
        return
    k = track_key(path)
    if not track_status.get(k):
        track_status[k] = True
        save_tracking_json()


def push_undo(path, boxes):
    undo_stack.append({"path": path, "boxes": copy_boxes(boxes)})
    while len(undo_stack) > UNDO_MAX:
        undo_stack.pop(0)


def reload_images():
    global all_image_list, undo_stack, dataset_clustered, cluster_groups, cluster_ordered_paths, crop_refs
    global size_filter_on, path_box_size, path_size_pct, size_ordered_paths, path_wh, crop_gen
    global box_size_cache, box_size_cache_ready
    all_image_list = list_images(IMAGES_DIR)
    undo_stack = []
    dataset_clustered = False
    cluster_groups = []
    cluster_ordered_paths = []
    crop_refs = []
    crop_gen = 0
    size_filter_on = False
    path_box_size = {}
    path_size_pct = {}
    size_ordered_paths = []
    box_size_cache = {}
    box_size_cache_ready = False
    path_wh = {}
    load_classes_from_dir(LABELS_DIR)
    apply_class_filter()
    if tracking_enabled:
        load_or_create_tracking()


def ensure_crop_refs():
    global crop_refs, crop_gen
    if crop_refs:
        return
    if sort_mode in ("size_asc", "size_desc") and not box_size_cache_ready:
        build_size_ranks()
    refs = []
    for path in image_list:
        if box_size_cache_ready and path in box_size_cache:
            boxes = [(bi, e) for bi, e in enumerate(box_size_cache[path])]
        else:
            boxes = [(bi, b) for bi, b in enumerate(load_boxes(path))]
        for bi, b in boxes:
            if class_filter_id is not None and int(b["cls"]) != int(class_filter_id):
                continue
            refs.append({"path": path, "box": b, "box_i": bi, "area": b.get("area", box_pixel_area(path, b))})
    if sort_mode == "size_desc":
        refs.sort(key=lambda r: r["area"], reverse=True)
    elif sort_mode == "size_asc":
        refs.sort(key=lambda r: r["area"])
    crop_gen += 1
    crop_refs = refs


def crop_jpg_response(bgr, quality=85):
    resp = send_file(encode_jpg(bgr, quality), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


def crop_patch(bgr, b, size=None):
    if size is None:
        size = CROP_THUMB
    h, w = bgr.shape[:2]
    bx1 = (b["cx"] - b["w"] / 2) * w
    by1 = (b["cy"] - b["h"] / 2) * h
    bx2 = (b["cx"] + b["w"] / 2) * w
    by2 = (b["cy"] + b["h"] / 2) * h
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    box_side = max(bw, bh)
    side = box_side * CROP_ZOOM
    img_min = float(min(w, h))
    if box_side < img_min * CROP_SMALL_FRAC:
        side = max(side, img_min * CROP_SMALL_VIEW)
    side = max(side, 32.0)
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    patch = bgr[y1:y2, x1:x2].copy()
    color = class_hex(b["cls"])
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    cv2.rectangle(
        patch,
        (int(round(bx1 - x1)), int(round(by1 - y1))),
        (int(round(bx2 - x1)), int(round(by2 - y1))),
        (rgb[2], rgb[1], rgb[0]), 2,
    )
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def state_payload():
    path = current_path()
    boxes = load_boxes(path) if path else []
    w = h = 1
    if path:
        bgr = load_bgr(path)
        h, w = bgr.shape[:2]
    names = []
    for i, p in enumerate(image_list):
        names.append({
            "i": i,
            "name": Path(p).name,
            "checked": bool(track_status.get(track_key(p), False)) if tracking_enabled else False,
        })
    return {
        "images_dir": IMAGES_DIR,
        "labels_dir": LABELS_DIR,
        "index": current_index,
        "total": len(image_list),
        "name": Path(path).name if path else "-",
        "path": path,
        "img_w": w,
        "img_h": h,
        "boxes": boxes,
        "classes": [class_name(i) for i in range(len(CLASS_NAMES))],
        "class_colors": [class_hex(i) for i in range(len(CLASS_NAMES))],
        "class_filter": class_filter_id,
        "sort_mode": sort_mode,
        "sort_by_cluster": sort_mode == "cluster",
        "size_filter_on": size_filter_on,
        "size_filter_lo": size_filter_lo,
        "size_filter_hi": size_filter_hi,
        "size_filter_ready": box_size_cache_ready,
        "dataset_clustered": dataset_clustered,
        "cluster_count": len(cluster_groups),
        "clusters": [{"i": i, "n": len(g)} for i, g in enumerate(cluster_groups)],
        "clustering": clustering,
        "cluster_msg": cluster_msg,
        "tracking": tracking_enabled,
        "place_w": PLACE_W,
        "place_h": PLACE_H,
        "box_thickness": BOX_THICKNESS,
        "label_size": LABEL_SIZE,
        "filter_params": FILTER_PARAMS,
        "nav": names,
        "checked": bool(track_status.get(track_key(path), False)) if (tracking_enabled and path) else False,
        "track_colors": {"yellow": TRACK_YELLOW, "blue": TRACK_BLUE},
    }


app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state_payload())


@app.route("/api/image")
def api_image():
    path = current_path()
    if not path:
        return ("", 404)
    bgr = load_bgr(path)
    filt = request.args.get("filter")
    if filt:
        bgr = apply_filter(bgr, filt)
    return send_file(encode_jpg(bgr), mimetype="image/jpeg")


def apply_folder(kind, d):
    global IMAGES_DIR, LABELS_DIR
    if kind == "images":
        IMAGES_DIR = d
        save_last_folders()
        reload_images()
    else:
        LABELS_DIR = d
        save_last_folders()
        load_classes_from_dir(LABELS_DIR)
        if image_list or all_image_list:
            apply_class_filter(current_path())
        else:
            reload_images()
    msg = None
    if tracking_enabled:
        msg = load_or_create_tracking()
    out = state_payload()
    out["track_info"] = msg
    return out


@app.route("/api/folders", methods=["POST"])
def api_folders():
    global IMAGES_DIR, LABELS_DIR
    data = request.get_json(force=True) or {}
    img = data.get("images_dir")
    lab = data.get("labels_dir")
    if img and os.path.isdir(img):
        IMAGES_DIR = img
    if lab and os.path.isdir(lab):
        LABELS_DIR = lab
    save_last_folders()
    reload_images()
    msg = None
    if tracking_enabled:
        msg = load_or_create_tracking()
    out = state_payload()
    out["track_info"] = msg
    return jsonify(out)


def pick_directory(title, initial):
    if not initial or not os.path.isdir(initial):
        initial = os.path.expanduser("~")
    if os.name != "nt":
        cmds = [
            ["zenity", "--file-selection", "--directory", "--title", title, "--filename", initial.rstrip("/") + "/"],
            ["kdialog", "--getexistingdirectory", initial, title],
        ]
        for args in cmds:
            try:
                r = subprocess.run(args, capture_output=True, timeout=600)
            except FileNotFoundError:
                continue
            except Exception:
                continue
            if r.returncode == 0:
                return (r.stdout or b"").decode("utf-8", "replace").strip()
            err = (r.stderr or b"").decode("utf-8", "replace")
            if r.returncode in (1, 5) and not err.strip():
                return ""
            continue
    env = os.environ.copy()
    env["LI_INITIAL"] = initial
    env["LI_TITLE"] = title
    code = (
        "import os,sys\n"
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root=tk.Tk(); root.withdraw()\n"
        "try:\n"
        " root.attributes('-topmost', True)\n"
        "except Exception:\n"
        " pass\n"
        "root.update()\n"
        "d=filedialog.askdirectory(initialdir=os.environ.get('LI_INITIAL') or os.path.expanduser('~'),"
        " title=os.environ.get('LI_TITLE') or 'Select folder')\n"
        "root.destroy()\n"
        "sys.stdout.buffer.write((d or '').encode('utf-8'))\n"
    )
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=600, **kw)
    except Exception:
        return None
    if r.returncode == 0:
        return (r.stdout or b"").decode("utf-8", "replace").strip()
    err = (r.stderr or b"").decode("utf-8", "replace")
    if "tkinter" in err.lower():
        return None
    return ""


@app.route("/api/browse", methods=["POST"])
def api_browse():
    kind = (request.get_json(force=True) or {}).get("kind")
    initial = IMAGES_DIR if kind == "images" else LABELS_DIR
    if not initial or not os.path.isdir(initial):
        initial = os.path.expanduser("~")
    title = "Select images folder" if kind == "images" else "Select annotations folder"
    try:
        d = pick_directory(title, initial)
    except Exception as e:
        return jsonify({"error": "Browse failed: %s" % e}), 500
    if d is None:
        return jsonify({
            "error": "No folder picker available. Paste the folder path in File and press Set. "
            "On Ubuntu you can also: sudo apt install zenity   or   sudo apt install python3-tk"
        }), 500
    if not d:
        return jsonify(state_payload())
    return jsonify(apply_folder(kind, d))


@app.route("/api/set_dir", methods=["POST"])
def api_set_dir():
    data = request.get_json(force=True) or {}
    kind = data.get("kind")
    path = (data.get("path") or "").strip().strip('"').strip("'")
    if kind not in ("images", "labels"):
        return jsonify({"error": "Invalid folder kind"}), 400
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Folder not found: %s" % path}), 400
    return jsonify(apply_folder(kind, os.path.abspath(path)))


@app.route("/api/goto", methods=["POST"])
def api_goto():
    global current_index
    data = request.get_json(force=True) or {}
    if "cluster" in data and dataset_clustered:
        ci = int(data["cluster"])
        if 0 <= ci < len(cluster_groups):
            for p in cluster_groups[ci]:
                if p in image_list:
                    current_index = image_list.index(p)
                    break
    elif "index" in data and image_list:
        current_index = max(0, min(len(image_list) - 1, int(data["index"])))
    return jsonify(state_payload())


@app.route("/api/boxes", methods=["POST"])
def api_boxes():
    path = current_path()
    if not path:
        return jsonify(state_payload())
    data = request.get_json(force=True) or {}
    boxes = data.get("boxes") or []
    old = load_boxes(path)
    if data.get("changed"):
        push_undo(path, old)
        mark_checked(path)
    write_boxes(path, boxes)
    return jsonify(state_payload())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    global current_index
    if not undo_stack:
        return jsonify(state_payload())
    entry = undo_stack.pop()
    write_boxes(entry["path"], entry["boxes"])
    if entry["path"] in image_list:
        current_index = image_list.index(entry["path"])
    return jsonify(state_payload())


@app.route("/api/delete", methods=["POST"])
def api_delete():
    global current_index, all_image_list, image_list, cluster_groups, cluster_ordered_paths, dataset_clustered, crop_refs
    path = current_path()
    if not path:
        return jsonify(state_payload())
    txt = label_path(path)
    try:
        if os.path.isfile(path):
            os.remove(path)
        if os.path.isfile(txt):
            os.remove(txt)
    except OSError as e:
        return jsonify({"error": str(e)}), 400
    if path in image_list:
        image_list.remove(path)
    if path in all_image_list:
        all_image_list.remove(path)
    box_size_cache.pop(path, None)
    path_box_size.pop(path, None)
    if box_size_cache_ready:
        rebuild_size_order()
    if tracking_enabled:
        track_status.pop(track_key(path), None)
        save_tracking_json()
    if dataset_clustered:
        cluster_groups = [[p for p in g if p != path] for g in cluster_groups]
        cluster_groups = [g for g in cluster_groups if g]
        cluster_ordered_paths = [p for g in cluster_groups for p in g]
        if not cluster_groups:
            dataset_clustered = False
    crop_refs = []
    if not image_list:
        current_index = -1
    else:
        current_index = min(current_index, len(image_list) - 1)
    return jsonify(state_payload())


@app.route("/api/filter_class", methods=["POST"])
def api_filter_class():
    global class_filter_id
    keep = current_path()
    class_filter_id = (request.get_json(force=True) or {}).get("cls")
    apply_class_filter(keep)
    return jsonify(state_payload())


@app.route("/api/size_filter/build", methods=["POST"])
def api_size_filter_build():
    global size_filter_on, size_filter_lo, size_filter_hi
    if not all_image_list:
        return jsonify({"error": "No images loaded"}), 400
    build_size_ranks()
    size_filter_on = True
    size_filter_lo = 0.0
    size_filter_hi = 1.0
    apply_class_filter(current_path())
    return jsonify(state_payload())


@app.route("/api/size_filter", methods=["POST"])
def api_size_filter():
    global size_filter_on, size_filter_lo, size_filter_hi
    data = request.get_json(force=True) or {}
    keep = current_path()
    if "enabled" in data:
        size_filter_on = bool(data["enabled"])
        if size_filter_on and not box_size_cache_ready:
            build_size_ranks()
    if "lo" in data:
        size_filter_lo = max(0.0, min(1.0, float(data["lo"])))
    if "hi" in data:
        size_filter_hi = max(0.0, min(1.0, float(data["hi"])))
    apply_class_filter(keep)
    return jsonify(state_payload())


@app.route("/api/sort", methods=["POST"])
def api_sort():
    global sort_mode
    mode = (request.get_json(force=True) or {}).get("mode") or "name"
    if mode not in ("name", "cluster", "size_asc", "size_desc"):
        return jsonify({"error": "Invalid sort mode"}), 400
    if mode == "cluster" and not dataset_clustered:
        return jsonify({"error": "Images are not clustered yet. Press Cluster Dataset first."}), 400
    if mode in ("size_asc", "size_desc") and not box_size_cache_ready:
        if not all_image_list:
            return jsonify({"error": "No images loaded"}), 400
        build_size_ranks()
    keep = current_path()
    sort_mode = mode
    apply_class_filter(keep)
    return jsonify(state_payload())


@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    global clustering, cluster_msg
    if clustering or not all_image_list:
        return jsonify(state_payload())
    clustering = True
    cluster_msg = "Clustering dataset... extracting features"
    paths = list(all_image_list)

    def work():
        global clustering, dataset_clustered, cluster_groups, cluster_ordered_paths, cluster_msg, crop_refs
        try:
            feats = np.zeros((len(paths), FEATURE_DIM), dtype=np.float32)
            with ThreadPoolExecutor(max_workers=max(1, CLUSTER_WORKERS)) as pool:
                for i, feat in enumerate(pool.map(extract_phash_feature, paths)):
                    if feat is not None:
                        feats[i] = feat
            assigned = np.zeros(len(paths), dtype=bool)
            groups = []
            r2 = CLUSTER_RADIUS * CLUSTER_RADIUS
            for seed in range(len(paths)):
                if assigned[seed]:
                    continue
                diff = feats - feats[seed]
                d2 = np.einsum("ij,ij->i", diff, diff)
                d2[assigned] = np.inf
                members = np.nonzero(d2 <= r2)[0]
                assigned[members] = True
                groups.append([paths[j] for j in members.tolist()])
            groups = sorted(groups, key=len, reverse=True)
            cluster_groups = groups
            cluster_ordered_paths = [p for g in groups for p in g]
            dataset_clustered = True
            crop_refs = []
            cluster_msg = "Clustered into %d groups" % len(groups)
        except Exception as e:
            cluster_msg = str(e)
        clustering = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(state_payload())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global PLACE_W, PLACE_H, BOX_THICKNESS, LABEL_SIZE
    data = request.get_json(force=True) or {}
    if "place_w" in data:
        PLACE_W = int(data["place_w"])
    if "place_h" in data:
        PLACE_H = int(data["place_h"])
    if "box_thickness" in data:
        BOX_THICKNESS = int(data["box_thickness"])
    if "label_size" in data:
        LABEL_SIZE = int(data["label_size"])
    if "filter_params" in data:
        for k, v in data["filter_params"].items():
            if k in FILTER_PARAMS and isinstance(v, dict):
                FILTER_PARAMS[k].update(v)
    return jsonify(state_payload())


@app.route("/api/classes", methods=["POST"])
def api_classes():
    global CLASS_NAMES
    names = (request.get_json(force=True) or {}).get("names") or []
    names = [str(n) for n in names]
    if names:
        CLASS_NAMES = names
        save_classes_txt()
    return jsonify(state_payload())


@app.route("/api/track", methods=["POST"])
def api_track():
    global tracking_enabled, track_status
    on = bool((request.get_json(force=True) or {}).get("enabled"))
    info = None
    if on:
        if not all_image_list:
            return jsonify({"error": "Load an images folder first, then enable Track Changes."}), 400
        tracking_enabled = True
        info = load_or_create_tracking()
    else:
        tracking_enabled = False
        track_status = {}
    out = state_payload()
    out["track_info"] = info
    return jsonify(out)


@app.route("/api/checked", methods=["POST"])
def api_checked():
    global current_index
    path = current_path()
    mark_checked(path)
    if image_list:
        current_index = (current_index + 1) % len(image_list)
    return jsonify(state_payload())


@app.route("/api/stats")
def api_stats():
    counts = [0] * len(CLASS_NAMES)
    n_box = 0
    for path in image_list:
        for b in load_boxes(path):
            n_box += 1
            i = int(b["cls"])
            if 0 <= i < len(counts):
                counts[i] += 1
    lines = ["Images: %d" % len(image_list), "Boxes: %d" % n_box, ""]
    for i, n in enumerate(counts):
        lines.append("%d %s: %d" % (i, class_name(i), n))
    return jsonify({"text": "\n".join(lines)})


@app.route("/api/crops")
def api_crops():
    ensure_crop_refs()
    offset = int(request.args.get("offset", 0))
    count = int(request.args.get("count", 100))
    end = min(offset + count, len(crop_refs))
    items = []
    for i in range(offset, end):
        r = crop_refs[i]
        b = r["box"]
        items.append({
            "i": i,
            "name": Path(r["path"]).name,
            "cls": b["cls"],
            "cls_name": class_name(b["cls"]),
            "color": class_hex(b["cls"]),
            "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"],
            "area": r.get("area", box_pixel_area(r["path"], b)),
        })
    return jsonify({
        "total": len(crop_refs),
        "offset": offset,
        "items": items,
        "gen": crop_gen,
        "sort_mode": sort_mode,
    })


@app.route("/api/crop_thumb/<int:idx>")
def api_crop_thumb(idx):
    ensure_crop_refs()
    g = request.args.get("g", type=int)
    if g is not None and g != crop_gen:
        return ("stale", 409)
    if idx < 0 or idx >= len(crop_refs):
        return ("", 404)
    r = crop_refs[idx]
    size = 720 if request.args.get("full") else CROP_THUMB
    patch = crop_patch(load_bgr(r["path"]), r["box"], size)
    return crop_jpg_response(patch, 85)


@app.route("/api/crop_image/<int:idx>")
def api_crop_image(idx):
    ensure_crop_refs()
    g = request.args.get("g", type=int)
    if g is not None and g != crop_gen:
        return ("stale", 409)
    if idx < 0 or idx >= len(crop_refs):
        return ("", 404)
    return crop_jpg_response(load_bgr(crop_refs[idx]["path"]), 90)


@app.route("/api/open_crop", methods=["POST"])
def api_open_crop():
    global current_index
    ensure_crop_refs()
    i = int((request.get_json(force=True) or {}).get("i", -1))
    if i < 0 or i >= len(crop_refs):
        return jsonify(state_payload())
    r = crop_refs[i]
    if r["path"] in image_list:
        current_index = image_list.index(r["path"])
    out = state_payload()
    out["select_box"] = r["box"]
    return jsonify(out)


load_last_folders()
os.makedirs(DATA_DIR, exist_ok=True)
reload_images()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
