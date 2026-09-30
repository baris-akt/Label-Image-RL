# LabelImage

Web app for viewing and editing **YOLO** bounding boxes. It has a single-image editor and a crop browser so you can scan boxes across a dataset, filter by class or size, and fix labels in place.

## Requirements

- Python 3.10+ (3.11 recommended)
- A local browser
- On Windows, folder browse uses Tkinter (included with the official Python installer)

Install:

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python app.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

Point **File → Browse Images** at the image folder and **File → Browse Annotations** at the YOLO `.txt` folder (`classes.txt` is read from there if present). Last folders are remembered locally and are not committed.

## Dataset layout

Images: `.jpg` `.jpeg` `.png` `.bmp` `.tif` `.tiff` `.webp`

Labels: one `name.txt` per image, YOLO lines:

```
class_id cx cy w h
```

`cx, cy, w, h` are normalized 0–1. Optional `classes.txt` lists class names, one per line.

## Single Image

- Draw, move, and resize boxes (**Edit E**), delete (**Remove R**), place a new box (**Place Box Q**)
- Double-click a box to change its class
- **Boxes (B)** / **Labels (L)** visibility
- **Crop Focus** zooms to the first box
- **Filter & Sort**: class filter, cluster sort (run **Cluster Dataset** first), box-size percentile sliders
- **Post Process**: gamma / CLAHE / unsharp / emboss preview
- **Track Changes**: mark images checked
- **Ctrl+Z** undo (last box edits)

Navigation does not wrap past the first or last image.

## Crops

Opens a grid of box crops (first 100; use **Show 100 / 500 / All**).

- Class filter shows **only that class’s boxes** (other boxes on the same image are hidden)
- Sort by name, cluster, **smallest→biggest**, or **biggest→smallest**
- Size ranks are cached after **Box Size Filtering** or the first size-sort; edits update the cache
- Click a crop to open the full image. Edit / remove / place / boxes / labels and class double-click work the same as on the single-image page

Box size is pixel area: `box_w × box_h × image_width × image_height`.

## Shortcuts

| Key | Action |
| --- | --- |
| E | Edit |
| R | Remove |
| Q | Place box |
| B | Toggle boxes |
| L | Toggle labels |
| F | Fit image |
| C | Image checked (tracking on) |
| Ctrl+Z | Undo |
| A / ← | Previous |
| D / → | Next |
| Esc | Close popup / modal |

## Project files

```
app.py                 # Flask app
templates/index.html
static/js/app.js
static/css/style.css
tools/cluster_images.py  # optional offline clustering helper
data/                    # local cache (gitignored)
```

The server binds to `127.0.0.1:5050` only.
