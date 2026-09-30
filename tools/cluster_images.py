"""Cluster all images in a folder (any names). Same feature + radius logic as before."""
from pathlib import Path
import os
import json
import csv
import re
import shutil

import numpy as np
from PIL import Image
import imagehash
from multiprocessing import Pool

# --- set these ---
source_roots = [
    r"C:\Users\aktas\Desktop\empty-body-images\cam6"
    r"C:\Users\aktas\Desktop\empty-body-images\cam7",
    r"C:\Users\aktas\Desktop\empty-body-images\cam8",
    r"C:\Users\aktas\Desktop\empty-body-images\cam9",
    r"C:\Users\aktas\Desktop\empty-body-images\cam10",
]
out_dir = r"C:\Users\aktas\Desktop\empty-body-images\clusters"
radius = 5.0
workers = 8
# "all" = copy every image into cluster folders
# "sample" = copy only evenly spaced samples
copy_mode = "all"
samples_per_cluster = 10
# -----------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PHASH_SIZE = 16
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
FEATURE_DIM = PHASH_BITS + 256


def extract_feature(path):
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


if __name__ == "__main__":
    paths = []
    print("scanning source folders...")
    for root in source_roots:
        root = Path(root)
        print(f"  {root}")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                p = Path(dirpath) / f
                if p.suffix.lower() in IMAGE_EXTS:
                    paths.append(p)

    paths = sorted(
        paths,
        key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))],
    )
    print(f"  total images: {len(paths)}")
    if not paths:
        raise SystemExit("no images found")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nextracting features for {len(paths)} images...")
    feats = np.zeros((len(paths), FEATURE_DIM), dtype=np.float32)
    with Pool(max(1, workers)) as pool:
        for i, feat in enumerate(pool.imap(extract_feature, [str(p) for p in paths], chunksize=16)):
            if feat is not None:
                feats[i] = feat
            if (i + 1) % 50 == 0 or i + 1 == len(paths):
                print(f"  features {i + 1}/{len(paths)}")

    np.savez_compressed(
        out / "features.npz",
        names=np.array([str(p) for p in paths]),
        feats=feats,
    )

    print(f"clustering with radius={radius}...")
    assigned = np.zeros(len(paths), dtype=bool)
    clusters = []
    r2 = radius * radius
    for seed in range(len(paths)):
        if assigned[seed]:
            continue
        diff = feats - feats[seed]
        d2 = np.einsum("ij,ij->i", diff, diff)
        d2[assigned] = np.inf
        members = np.nonzero(d2 <= r2)[0]
        assigned[members] = True
        clusters.append(members.tolist())

    clusters = sorted(clusters, key=len, reverse=True)

    clusters_json = {}
    row_to_cluster = {}
    row_is_rep = set()
    for ci, members in enumerate(clusters, start=1):
        cid = f"C{ci:04d}"
        folder_name = f"cluster_{ci:03d}"
        cluster_dir = out / folder_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        if len(members) == 1:
            rep = members[0]
        else:
            sub = feats[members]
            centroid = sub.mean(axis=0)
            d2 = np.einsum("ij,ij->i", (sub - centroid), (sub - centroid))
            rep = members[int(np.argmin(d2))]
        row_is_rep.add(rep)

        member_paths = [paths[r] for r in members]
        if copy_mode == "sample":
            step = max(1, len(member_paths) // samples_per_cluster)
            to_copy = member_paths[::step][:samples_per_cluster]
            dest_dir = cluster_dir / "samples"
        else:
            to_copy = member_paths
            dest_dir = cluster_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        used_names = set()
        for src in to_copy:
            name = src.name
            if name in used_names:
                name = f"{src.parent.name}_{src.name}"
            used_names.add(name)
            shutil.copy2(src, dest_dir / name)

        print(f"  {folder_name}: copied {len(to_copy)} / {len(member_paths)}")

        clusters_json[cid] = {
            "folder": folder_name,
            "representative": str(paths[rep]),
            "members": [str(p) for p in member_paths],
            "size": len(members),
        }
        for r in members:
            row_to_cluster[r] = cid

    (out / "clusters.json").write_text(json.dumps(clusters_json, indent=2), encoding="utf-8")

    with (out / "representatives.txt").open("w", encoding="utf-8") as fh:
        for info in clusters_json.values():
            fh.write(info["representative"] + "\n")

    with (out / "dedup_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "cluster_id", "is_representative"])
        for r in sorted(row_to_cluster):
            w.writerow([str(paths[r]), row_to_cluster[r], "true" if r in row_is_rep else "false"])

    sizes = [c["size"] for c in clusters_json.values()]
    summary = {
        "total_frames": len(paths),
        "cluster_count": len(clusters_json),
        "singleton_clusters": sum(1 for s in sizes if s == 1),
        "largest_cluster": max(sizes) if sizes else 0,
        "mean_cluster_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
        "params": {"feature": "phash16+gray_hist256", "radius": radius, "copy_mode": copy_mode},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{summary['cluster_count']} clusters from {summary['total_frames']} images -> {out}")
    print("done")
