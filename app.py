#!/usr/bin/env python3
"""YOLO label web app. Combined viewer + crop search + change tracking.

    python app.py
    python3 app.py

Browse folders in the UI (File > Browse Images / Browse Annotations),
or paste a folder path and press Set.
"""
import os
import io
import re
import html
import sys
import json
import threading
import subprocess
import colorsys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import imagehash

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRACK_CACHE_DIR = os.path.join(DATA_DIR, "tracking_cache")

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

def opposite_hex(cls_idx):
    s, v = class_sv(cls_idx)
    h = ((HUE_ORDER[int(cls_idx) % HUE_BINS] + HUE_BINS // 2) % HUE_BINS) / float(HUE_BINS)
    r, g, b = colorsys.hsv_to_rgb(h, float(s), float(v))
    return "#%02x%02x%02x" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PLACE_W, PLACE_H = 50, 50
BOX_THICKNESS, LABEL_SIZE = 2, 10
UNDO_MAX = 10
CLUSTER_CROP = 224
CLUSTER_MAX_BITS = 14
CLUSTER_MERGE_BITS = 14
CLUSTER_ATTACH_BITS = 22
CLUSTER_SMALL = 2
PHASH_SIZE = 8
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
FEATURE_DIM = PHASH_BITS
CLUSTER_WORKERS = max(2, min((os.cpu_count() or 8) - 5, 12))
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
images_no_txt = 0
images_empty_txt = 0
images_scanned = 0
folder_busy = False
base_boxes = {}
touched_paths = set()
deleted_paths = set()
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
path_classes = {}
path_wh = {}
crop_gen = 0
indexing = False
index_gen = 0
index_cur = 0
index_total = 0
index_msg = ""
index_error = ""
INDEX_WORKERS = max(4, (os.cpu_count() or 8) - 5)
dataset_clustered = False
box_cluster = {}
cluster_groups = []
cluster_ordered_paths = []
clustering = False
cluster_msg = ""
cluster_pct = 0.0
undo_stack = []
tracking_enabled = False
track_status = {}
crop_refs = []
crop_cluster_id = None
observers = []
OBS_COLORS = ["#ffff00", "#ff00ff", "#00ffff", "#00ff00", "#0000ff"]
folder_image_list = []
skip_obs_only = True


def class_name(cls_idx):
    i = int(cls_idx)
    if 0 <= i < len(CLASS_NAMES) and str(CLASS_NAMES[i]).strip():
        return str(CLASS_NAMES[i]).strip()
    return "Unnamed-Cls%d" % i


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
    if box_size_cache_ready and img_path in box_size_cache:
        return [{
            "cls": e["cls"], "cx": e["cx"], "cy": e["cy"], "w": e["w"], "h": e["h"]
        } for e in box_size_cache[img_path]]
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


def drop_from_lists(path):
    global current_index, all_image_list, image_list, cluster_groups, cluster_ordered_paths, dataset_clustered, crop_refs
    if path in image_list:
        image_list.remove(path)
    if path in all_image_list:
        all_image_list.remove(path)
    box_size_cache.pop(path, None)
    path_box_size.pop(path, None)
    path_classes.pop(path, None)
    path_wh.pop(path, None)
    if box_size_cache_ready:
        rebuild_size_order()
    if dataset_clustered:
        for k in [k for k in box_cluster if k[0] == path]:
            del box_cluster[k]
        rebuild_cluster_order()
    crop_refs = []
    if not image_list:
        current_index = -1
    else:
        current_index = min(current_index, len(image_list) - 1)


def write_boxes(path, boxes):
    global crop_refs
    p = label_path(path)
    with open(p, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (b["cls"], b["cx"], b["cy"], b["w"], b["h"]))
    crop_refs = []
    touched_paths.add(path)
    if not boxes and (skip_obs_only or not has_observed(path)):
        drop_from_lists(path)
        return
    if dataset_clustered:
        old_ci = {k[1]: box_cluster.pop(k) for k in [k for k in box_cluster if k[0] == path]}
        fallback = next(iter(old_ci.values()), None)
        left_new = []
        for b in boxes:
            nk = box_key(b)
            if nk in old_ci:
                box_cluster[(path, nk)] = old_ci.pop(nk)
            else:
                left_new.append(nk)
        left_old = list(old_ci.values())
        for nk in left_new:
            ci = left_old.pop(0) if left_old else fallback
            if ci is not None:
                box_cluster[(path, nk)] = ci
        rebuild_cluster_order()
    if box_size_cache_ready:
        cache_boxes_for_path(path, boxes)
        path_classes[path] = {int(b["cls"]) for b in boxes}
        rebuild_size_order()


def copy_boxes(boxes):
    return [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"]} for b in boxes]


def image_has_class(path, cls_id):
    cid = int(cls_id)
    if box_size_cache_ready and path in path_classes:
        return cid in path_classes[path]
    for b in load_boxes(path):
        if int(b["cls"]) == cid:
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
    if box_size_cache_ready:
        return
    for p in all_image_list:
        boxes = load_boxes(p) if p not in box_size_cache else None
        if boxes is not None:
            cache_boxes_for_path(p, boxes)
            path_classes[p] = {int(b["cls"]) for b in boxes}
        elif p not in path_classes:
            path_classes[p] = {int(b["cls"]) for b in box_size_cache.get(p, [])}
    rebuild_size_order()
    box_size_cache_ready = True


def clear_dataset_index():
    global box_size_cache, box_size_cache_ready, path_box_size, path_size_pct, size_ordered_paths
    global path_wh, path_classes, crop_refs, crop_gen, index_cur, index_total, index_msg, index_error, index_gen
    index_gen += 1
    box_size_cache = {}
    box_size_cache_ready = False
    path_box_size = {}
    path_size_pct = {}
    size_ordered_paths = []
    path_wh = {}
    path_classes = {}
    crop_refs = []
    crop_gen = 0
    index_cur = 0
    index_total = 0
    index_msg = ""
    index_error = ""


def read_yolo(txt):
    boxes = []
    if not os.path.isfile(txt):
        return None
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append({"cls": max(0, int(float(parts[0]))), "cx": float(parts[1]),
                          "cy": float(parts[2]), "w": float(parts[3]), "h": float(parts[4])})
    return boxes


def has_observed(path):
    return any(o["boxes"].get(path) for o in observers)


def keep_image(path):
    return bool(box_size_cache.get(path)) or (not skip_obs_only and has_observed(path))


def scan_observers(obs_list, paths, my_gen):
    global index_cur, index_total, index_msg
    for o in obs_list:
        o["boxes"] = {}
        o["n_txt"] = 0
    if not obs_list or not paths:
        return
    index_cur = 0
    index_total = len(paths)
    index_msg = "Scanning observed annotations..."

    def scan_one(p):
        stem = Path(p).stem + ".txt"
        return p, [read_yolo(os.path.join(o["dir"], stem)) for o in obs_list]

    done = 0
    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as ex:
        for p, found in ex.map(scan_one, paths):
            if my_gen != index_gen:
                return
            for o, boxes in zip(obs_list, found):
                if boxes is not None:
                    o["n_txt"] += 1
                    if boxes:
                        o["boxes"][p] = boxes
            done += 1
            index_cur = done
            if done == 1 or done % 25 == 0 or done == index_total:
                index_msg = "Scanning observed annotations... %d / %d" % (done, index_total)


def rebuild_visible_images():
    global all_image_list
    all_image_list = [p for p in folder_image_list if keep_image(p)]
    for p in all_image_list:
        base_boxes.setdefault(p, [box_key(b) for b in box_size_cache.get(p, [])])
    rebuild_size_order()
    apply_class_filter(current_path())
    if dataset_clustered:
        rebuild_cluster_order()


def start_observed_scan():
    global indexing, index_error, index_gen, index_msg
    index_gen += 1
    my_gen = index_gen
    indexing = True
    index_error = ""
    index_msg = "Scanning observed annotations..."

    def work():
        global indexing, index_error, index_msg, crop_refs
        try:
            scan_observers(observers, list(folder_image_list), my_gen)
            if my_gen != index_gen:
                return
            rebuild_visible_images()
            crop_refs = []
            index_msg = "Scan finished"
        except Exception as e:
            if my_gen == index_gen:
                index_error = str(e)
                index_msg = "Observed scan failed"
        finally:
            if my_gen == index_gen:
                indexing = False

    threading.Thread(target=work, daemon=True).start()


def folders_ready():
    return bool(IMAGES_DIR and LABELS_DIR and os.path.isdir(IMAGES_DIR) and os.path.isdir(LABELS_DIR) and all_image_list)


def start_dataset_index():
    global indexing, index_msg, index_error, index_cur, index_total, index_gen
    if not folders_ready():
        return False
    if box_size_cache_ready:
        return False
    index_gen += 1
    my_gen = index_gen
    indexing = True
    index_error = ""
    index_cur = 0
    index_total = len(all_image_list)
    index_msg = "Starting dataset scan..."
    paths = list(all_image_list)
    labels_dir = LABELS_DIR
    n_workers = INDEX_WORKERS

    def scan_one(p):
        boxes = []
        lp = os.path.join(labels_dir, Path(p).stem + ".txt")
        has_txt = os.path.isfile(lp)
        if has_txt:
            with open(lp, "r", encoding="utf-8") as f:
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
        entries = []
        mx = 0.0
        wh = None
        if boxes:
            try:
                with Image.open(p) as im:
                    wh = im.size
            except Exception:
                wh = (1, 1)
            iw, ih = wh
            for b in boxes:
                area = float(b["w"]) * float(b["h"]) * float(iw) * float(ih)
                entries.append({
                    "cls": b["cls"], "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"], "area": area
                })
                if area > mx:
                    mx = area
        return p, entries, mx, {int(b["cls"]) for b in boxes}, wh, has_txt

    def work():
        global indexing, box_size_cache_ready, index_cur, index_total, index_msg, index_error, crop_refs
        global all_image_list, images_no_txt, images_empty_txt, images_scanned
        try:
            index_total = len(paths)
            done = 0
            no_txt = 0
            empty_txt = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(scan_one, p) for p in paths]
                for fut in as_completed(futs):
                    if my_gen != index_gen:
                        break
                    p, entries, mx, classes, wh, has_txt = fut.result()
                    if not has_txt:
                        no_txt += 1
                    elif not entries:
                        empty_txt += 1
                    box_size_cache[p] = entries
                    path_box_size[p] = mx
                    path_classes[p] = classes
                    if wh is not None:
                        path_wh[p] = wh
                    done += 1
                    index_cur = done
                    if done == 1 or done % 25 == 0 or done == index_total:
                        index_msg = "Scanning labels and box sizes... %d / %d" % (done, index_total)
            if my_gen != index_gen:
                return
            images_scanned = len(paths)
            images_no_txt = no_txt
            images_empty_txt = empty_txt
            if observers:
                scan_observers(observers, paths, my_gen)
                if my_gen != index_gen:
                    return
            all_image_list = [p for p in paths if keep_image(p)]
            base_boxes.clear()
            for p in all_image_list:
                base_boxes[p] = [box_key(b) for b in box_size_cache.get(p, [])]
            touched_paths.clear()
            deleted_paths.clear()
            rebuild_size_order()
            box_size_cache_ready = True
            apply_class_filter()
            crop_refs = []
            index_msg = "Scan finished"
        except Exception as e:
            if my_gen == index_gen:
                index_error = str(e)
                index_msg = "Dataset index failed"
                box_size_cache_ready = False
        finally:
            if my_gen == index_gen:
                indexing = False

    threading.Thread(target=work, daemon=True).start()
    return True


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
    snap_crop_cluster()


def rebuild_cluster_order():
    global cluster_groups, cluster_ordered_paths, dataset_clustered
    n = (max(box_cluster.values()) + 1) if box_cluster else 0
    groups = [[] for _ in range(n)]
    for k, ci in box_cluster.items():
        groups[ci].append(k)
    cluster_groups = groups
    seen = set()
    order = []
    for g in groups:
        for p, _ in g:
            if p not in seen:
                seen.add(p)
                order.append(p)
    order += [p for p in all_image_list if p not in seen]
    cluster_ordered_paths = order
    dataset_clustered = bool(box_cluster)


def visible_clusters():
    shown = set(image_list)
    counts = [0] * len(cluster_groups)
    for (p, k), ci in box_cluster.items():
        if p in shown and (class_filter_id is None or k[0] == int(class_filter_id)):
            counts[ci] += 1
    return [{"i": i, "n": n} for i, n in enumerate(counts) if n]


def first_image_of_cluster(ci):
    allow = {p for p, _ in cluster_groups[ci]}
    for i, p in enumerate(image_list):
        if p in allow:
            return i
    return None


def snap_crop_cluster():
    global crop_cluster_id
    if crop_cluster_id is None or sort_mode != "cluster" or not cluster_groups:
        return
    vis = [c["i"] for c in visible_clusters()]
    if not vis:
        crop_cluster_id = None
        return
    if crop_cluster_id in vis:
        return
    later = [i for i in vis if i > crop_cluster_id]
    crop_cluster_id = later[0] if later else vis[0]


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


def extract_box_features(job):
    path, boxes = job
    bgr = load_bgr(path)
    h, w = bgr.shape[:2]
    half = CLUSTER_CROP // 2
    out = []
    for b in boxes:
        cx = min(max(int(round(float(b["cx"]) * w)), 0), w - 1)
        cy = min(max(int(round(float(b["cy"]) * h)), 0), h - 1)
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + CLUSTER_CROP, y1 + CLUSTER_CROP
        crop = bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        crop = cv2.copyMakeBorder(crop, max(0, -y1), max(0, y2 - h), max(0, -x1), max(0, x2 - w),
                                  cv2.BORDER_REPLICATE)
        gray = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        ph = imagehash.phash(Image.fromarray(gray), hash_size=PHASH_SIZE)
        out.append(np.asarray(ph.hash, dtype=np.float32).flatten())
    return out


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
    global box_size_cache, box_size_cache_ready, crop_cluster_id, images_no_txt, images_empty_txt, images_scanned
    global folder_image_list
    all_image_list = list_images(IMAGES_DIR)
    folder_image_list = list(all_image_list)
    images_no_txt = 0
    images_empty_txt = 0
    images_scanned = 0
    base_boxes.clear()
    touched_paths.clear()
    deleted_paths.clear()
    undo_stack = []
    dataset_clustered = False
    box_cluster.clear()
    cluster_groups = []
    cluster_ordered_paths = []
    size_filter_on = False
    crop_cluster_id = None
    clear_dataset_index()
    load_classes_from_dir(LABELS_DIR)
    apply_class_filter()
    if tracking_enabled:
        load_or_create_tracking()
    if folders_ready():
        start_dataset_index()


def ensure_crop_refs():
    global crop_refs, crop_gen
    if crop_refs:
        return
    need_area = sort_mode in ("size_asc", "size_desc")
    if need_area and not box_size_cache_ready:
        build_size_ranks()
    paths = list(image_list)
    by_cluster = sort_mode == "cluster" and dataset_clustered
    want_ci = int(crop_cluster_id) if (by_cluster and crop_cluster_id is not None) else None
    refs = []
    for path in paths:
        if box_size_cache_ready and path in box_size_cache:
            boxes = [(bi, e) for bi, e in enumerate(box_size_cache[path])]
        else:
            boxes = [(bi, b) for bi, b in enumerate(load_boxes(path))]
        for bi, b in boxes:
            if class_filter_id is not None and int(b["cls"]) != int(class_filter_id):
                continue
            ci = box_cluster.get((path, box_key(b))) if by_cluster else None
            if want_ci is not None and ci != want_ci:
                continue
            area = b.get("area")
            if need_area and area is None:
                area = box_pixel_area(path, b)
            refs.append({"path": path, "box": b, "box_i": bi, "area": area if area is not None else 0.0,
                         "ci": ci if ci is not None else 10 ** 9})
    if by_cluster:
        refs.sort(key=lambda r: r["ci"])
    elif sort_mode == "size_desc":
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
        "observers": [{"id": o["id"], "name": o["name"], "dir": o["dir"], "color": o["color"],
                       "color2": o["color2"]} for o in observers],
        "skip_obs_only": skip_obs_only,
        "obs_boxes": [{"id": o["id"], "boxes": o["boxes"].get(path, [])} for o in observers] if path else [],
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
        "index_ready": box_size_cache_ready,
        "indexing": indexing,
        "index_cur": index_cur,
        "index_total": index_total,
        "index_msg": index_msg,
        "index_error": index_error,
        "dataset_clustered": dataset_clustered,
        "cluster_count": len(cluster_groups),
        "clusters": visible_clusters(),
        "clustering": clustering,
        "cluster_msg": cluster_msg,
        "cluster_pct": cluster_pct,
        "crop_cluster_id": crop_cluster_id,
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
    global IMAGES_DIR, LABELS_DIR, folder_busy
    folder_busy = True
    try:
        if kind == "images":
            IMAGES_DIR = d
            LABELS_DIR = ""
            observers.clear()
        else:
            LABELS_DIR = d
        reload_images()
        msg = None
        if tracking_enabled:
            msg = load_or_create_tracking()
    finally:
        folder_busy = False
    out = state_payload()
    out["track_info"] = msg
    return out


def busy_error():
    if folder_busy:
        return "Still loading a folder. Please wait."
    if indexing:
        return "Dataset scan in progress. Please wait."
    if clustering:
        return "Clustering in progress. Please wait."
    return None


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
    err = busy_error()
    if err:
        return jsonify({"error": err}), 409
    data = request.get_json(force=True) or {}
    kind = data.get("kind")
    initial = (data.get("initial") or "").strip()
    if not initial or not os.path.isdir(initial):
        initial = IMAGES_DIR if kind == "images" else LABELS_DIR
    if not initial or not os.path.isdir(initial):
        initial = os.path.expanduser("~")
    if kind == "images":
        title = "Select images folder"
    elif kind == "observed":
        title = "Select observed annotations folder"
    else:
        title = "Select annotations folder"
    try:
        d = pick_directory(title, initial)
    except Exception as e:
        return jsonify({"error": "Browse failed: %s" % e}), 500
    if d is None:
        return jsonify({
            "error": "No folder picker available. Paste the folder path in File and press Set. "
            "On Ubuntu you can also: sudo apt install zenity   or   sudo apt install python3-tk"
        }), 500
    return jsonify({"path": d or ""})


@app.route("/api/set_dir", methods=["POST"])
def api_set_dir():
    err = busy_error()
    if err:
        return jsonify({"error": err}), 409
    data = request.get_json(force=True) or {}
    kind = data.get("kind")
    path = (data.get("path") or "").strip().strip('"').strip("'")
    if kind not in ("images", "labels"):
        return jsonify({"error": "Invalid folder kind"}), 400
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Folder not found: %s" % path}), 400
    return jsonify(apply_folder(kind, os.path.abspath(path)))


@app.route("/api/observed/add", methods=["POST"])
def api_observed_add():
    err = busy_error()
    if err:
        return jsonify({"error": err}), 409
    if not box_size_cache_ready or not folder_image_list:
        return jsonify({"error": "Load images and annotations folders first and wait for the scan."}), 400
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    d = (data.get("dir") or "").strip().strip('"').strip("'")
    if not name:
        return jsonify({"error": "Enter a text for the observed annotations."}), 400
    if not d or not os.path.isdir(d):
        return jsonify({"error": "Folder not found: %s" % d}), 400
    used = {o["slot"] for o in observers}
    slot = 0
    while slot in used:
        slot += 1
    observers.append({
        "id": max([o["id"] for o in observers] + [-1]) + 1,
        "slot": slot,
        "name": name,
        "dir": os.path.abspath(d),
        "color": OBS_COLORS[slot % len(OBS_COLORS)],
        "color2": OBS_COLORS[slot % len(OBS_COLORS)],
        "boxes": {},
        "n_txt": 0,
    })
    start_observed_scan()
    return jsonify(state_payload())


@app.route("/api/observed/skip", methods=["POST"])
def api_observed_skip():
    global skip_obs_only, crop_refs
    err = busy_error()
    if err:
        return jsonify({"error": err}), 409
    skip_obs_only = bool((request.get_json(force=True) or {}).get("skip", True))
    if box_size_cache_ready:
        rebuild_visible_images()
    crop_refs = []
    return jsonify(state_payload())


@app.route("/api/observed/remove", methods=["POST"])
def api_observed_remove():
    global crop_refs
    err = busy_error()
    if err:
        return jsonify({"error": err}), 409
    oid = (request.get_json(force=True) or {}).get("id")
    observers[:] = [o for o in observers if o["id"] != oid]
    if box_size_cache_ready:
        rebuild_visible_images()
    crop_refs = []
    return jsonify(state_payload())


@app.route("/api/index/build", methods=["POST"])
def api_index_build():
    if not folders_ready():
        return jsonify({"error": "Set both images and annotations folders first."}), 400
    if box_size_cache_ready:
        return jsonify(state_payload())
    start_dataset_index()
    return jsonify(state_payload())


@app.route("/api/index/status")
def api_index_status():
    return jsonify({
        "indexing": indexing,
        "index_ready": box_size_cache_ready,
        "index_cur": index_cur,
        "index_total": index_total,
        "index_msg": index_msg,
        "index_error": index_error,
        "pct": (100.0 * index_cur / index_total) if index_total else (100.0 if box_size_cache_ready else 0.0),
    })


@app.route("/api/goto", methods=["POST"])
def api_goto():
    global current_index
    data = request.get_json(force=True) or {}
    if "cluster" in data and dataset_clustered:
        ci = int(data["cluster"])
        vis = [c["i"] for c in visible_clusters()]
        if vis and ci not in vis:
            later = [i for i in vis if i > ci]
            ci = later[0] if later else vis[0]
        if 0 <= ci < len(cluster_groups):
            i = first_image_of_cluster(ci)
            if i is not None:
                current_index = i
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
    drop_from_lists(path)
    if path in folder_image_list:
        folder_image_list.remove(path)
    for o in observers:
        o["boxes"].pop(path, None)
    deleted_paths.add(path)
    touched_paths.add(path)
    if tracking_enabled:
        track_status.pop(track_key(path), None)
        save_tracking_json()
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
    global sort_mode, crop_cluster_id, crop_refs
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
    if mode == "cluster" and dataset_clustered:
        if crop_cluster_id is None or not (0 <= int(crop_cluster_id) < len(cluster_groups)):
            crop_cluster_id = 0
    else:
        crop_cluster_id = None
    crop_refs = []
    apply_class_filter(keep)
    return jsonify(state_payload())


@app.route("/api/crop_cluster", methods=["POST"])
def api_crop_cluster():
    global crop_cluster_id, crop_refs, current_index
    data = request.get_json(force=True) or {}
    if "cluster" in data and data.get("cluster") is None:
        crop_cluster_id = None
        crop_refs = []
        return jsonify(state_payload())
    if not dataset_clustered or not cluster_groups:
        return jsonify({"error": "Images are not clustered yet."}), 400
    if "cluster" not in data:
        return jsonify({"error": "cluster required"}), 400
    ci = int(data["cluster"])
    if ci < 0 or ci >= len(cluster_groups):
        return jsonify({"error": "Invalid cluster"}), 400
    vis = [c["i"] for c in visible_clusters()]
    if vis and ci not in vis:
        later = [i for i in vis if i > ci]
        ci = later[0] if later else vis[0]
    crop_cluster_id = ci
    crop_refs = []
    i = first_image_of_cluster(ci)
    if i is not None:
        current_index = i
    return jsonify(state_payload())


@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    global clustering, cluster_msg, cluster_pct
    if clustering or not all_image_list:
        return jsonify(state_payload())
    clustering = True
    cluster_pct = 0.0
    cluster_msg = "Clustering dataset... extracting box features"
    jobs = [(p, box_size_cache.get(p) or load_boxes(p)) for p in all_image_list]

    def work():
        global clustering, cluster_msg, crop_refs, cluster_pct
        try:
            keys = []
            feats = []
            n_img = len(jobs)
            with ThreadPoolExecutor(max_workers=max(1, CLUSTER_WORKERS)) as pool:
                for i, (job, fl) in enumerate(zip(jobs, pool.map(extract_box_features, jobs))):
                    for b, f in zip(job[1], fl):
                        keys.append((job[0], box_key(b)))
                        feats.append(f)
                    cluster_pct = 70.0 * (i + 1) / float(n_img)
                    cluster_msg = "Extracting box features... %d / %d images" % (i + 1, n_img)
            n = len(keys)
            feats = np.array(feats, dtype=np.float32).reshape(n, FEATURE_DIM)
            assigned = np.zeros(n, dtype=bool)
            groups = []
            cluster_msg = "Grouping similar boxes..."
            for seed in range(n):
                if assigned[seed]:
                    continue
                d = np.abs(feats - feats[seed]).sum(axis=1)
                d[assigned] = np.inf
                members = np.nonzero(d <= CLUSTER_MAX_BITS)[0]
                assigned[members] = True
                groups.append(members.tolist())
                cluster_pct = 70.0 + 25.0 * float(assigned.sum()) / float(n)
                cluster_msg = "Grouping similar boxes... %d / %d" % (int(assigned.sum()), n)
            cluster_msg = "Merging similar clusters..."
            groups = sorted(groups, key=len, reverse=True)
            cents = np.array([feats[g].mean(axis=0) for g in groups], dtype=np.float32).reshape(-1, FEATURE_DIM)
            k = len(groups)
            owner = np.arange(k)
            merged = np.zeros(k, dtype=bool)
            for a in range(k):
                if merged[a]:
                    continue
                d = np.abs(cents - cents[a]).sum(axis=1)
                d[merged] = np.inf
                d[a] = np.inf
                near = np.nonzero(d <= CLUSTER_MERGE_BITS)[0]
                owner[near] = a
                merged[near] = True
            sizes = np.zeros(k, dtype=np.int64)
            for a in range(k):
                sizes[owner[a]] += len(groups[a])
            big = np.nonzero((~merged) & (sizes > CLUSTER_SMALL))[0]
            if len(big):
                for a in np.nonzero((~merged) & (sizes <= CLUSTER_SMALL))[0]:
                    d = np.abs(cents[big] - cents[a]).sum(axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= CLUSTER_ATTACH_BITS:
                        owner[owner == a] = big[j]
            final = {}
            for a in range(k):
                final.setdefault(int(owner[a]), []).extend(groups[a])
            groups = sorted(final.values(), key=len, reverse=True)
            box_cluster.clear()
            for ci, g in enumerate(groups):
                for j in g:
                    box_cluster[keys[j]] = ci
            rebuild_cluster_order()
            apply_class_filter(current_path())
            crop_refs = []
            cluster_pct = 100.0
            cluster_msg = "Clustered %d boxes into %d groups" % (n, len(groups))
        except Exception as e:
            cluster_msg = str(e)
            cluster_pct = 0.0
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


def box_key(b):
    return (int(b["cls"]), round(float(b["cx"]), 6), round(float(b["cy"]), 6),
            round(float(b["w"]), 6), round(float(b["h"]), 6))


def diff_boxes(old, new, edited, deleted, added):
    rest_new = list(new)
    rest_old = []
    for k in old:
        if k in rest_new:
            rest_new.remove(k)
        else:
            rest_old.append(k)
    for k in list(rest_old):
        same = [n for n in rest_new if n[0] == k[0]]
        if same:
            rest_new.remove(same[0])
            rest_old.remove(k)
            edited[k[0]] = edited.get(k[0], 0) + 1
    while rest_old and rest_new:
        rest_old.pop(0)
        n = rest_new.pop(0)
        edited[n[0]] = edited.get(n[0], 0) + 1
    for k in rest_old:
        deleted[k[0]] = deleted.get(k[0], 0) + 1
    for n in rest_new:
        added[n[0]] = added.get(n[0], 0) + 1


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
    tracked = bool(images_scanned and touched_paths)
    edited, deleted, added = {}, {}, {}
    edited_imgs = 0
    for p in touched_paths:
        if p not in base_boxes:
            continue
        if p in deleted_paths:
            new = []
        else:
            new = [box_key(b) for b in box_size_cache.get(p, [])]
        if p not in deleted_paths and sorted(new) != sorted(base_boxes[p]):
            edited_imgs += 1
        diff_boxes(base_boxes[p], new, edited, deleted, added)
    n_del_imgs = len(deleted_paths)

    def parts(items):
        shown = [(n, label, color) for n, label, color in items if n]
        if not tracked or not shown:
            return ""
        return "  -->  " + ", ".join("<span style='color:%s'>%d %s</span>" % (color, n, label)
                                     for n, label, color in shown)

    def chg(c=None):
        if c is None:
            e, d, a = sum(edited.values()), sum(deleted.values()), sum(added.values())
        else:
            e, d, a = edited.get(c, 0), deleted.get(c, 0), added.get(c, 0)
        return parts([(e, "edited box", "#f5c542"), (d, "deleted box", "#ff5c5c"), (a, "added box", "#4cd964")])

    lines = []
    if images_scanned:
        folder_now = images_scanned - n_del_imgs
        with_boxes = sum(1 for p in all_image_list if box_size_cache.get(p))
        img_chg = parts([(edited_imgs, "edited image", "#f5c542"), (n_del_imgs, "deleted image", "#ff5c5c")])
        lines += [
            "Images in folder: %d%s" % (folder_now, img_chg),
            "Images with boxes: %d" % with_boxes,
            "Images with no txt: %d" % images_no_txt,
            "Images with empty txt (no boxes): %d" % max(0, folder_now - images_no_txt - with_boxes),
        ]
        if observers:
            obs_only = sum(1 for p in folder_image_list if not box_size_cache.get(p) and has_observed(p))
            lines.append("Images with only observed boxes: %d (%s)" % (obs_only, "skipped" if skip_obs_only else "shown"))
        lines.append("")
    lines += [
        "Images shown: %d%s" % (len(image_list), parts([(n_del_imgs, "deleted", "#ff5c5c")])),
        "Total boxes: %d%s" % (n_box, chg()),
        "",
    ]
    for i, n in enumerate(counts):
        lines.append("%d %s: %d%s" % (i, html.escape(class_name(i)), n, chg(i)))
    for o in observers:
        oc = [0] * len(CLASS_NAMES)
        n_obs = 0
        for p, bl in o["boxes"].items():
            for b in bl:
                n_obs += 1
                if 0 <= int(b["cls"]) < len(oc):
                    oc[int(b["cls"])] += 1
        lines += [
            "",
            "<span style='color:%s'>Observed \"%s\"</span> (%s):" % (o["color"], html.escape(o["name"]), html.escape(o["dir"])),
            "  Images with txt: %d   Images with boxes: %d   Total boxes: %d" % (o["n_txt"], len(o["boxes"]), n_obs),
        ]
        for i, n in enumerate(oc):
            lines.append("  %d %s: %d" % (i, html.escape(class_name(i)), n))
    out = "\n".join(lines)
    return jsonify({"html": out, "text": re.sub(r"<[^>]+>", "", out)})


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
            "area": r.get("area", 0.0),
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
    bgr = load_bgr(crop_refs[idx]["path"])
    filt = request.args.get("filter")
    if filt:
        bgr = apply_filter(bgr, filt)
    return crop_jpg_response(bgr, 90)


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


os.makedirs(DATA_DIR, exist_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
