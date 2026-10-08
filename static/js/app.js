let S = null;
let mode = "view";
let selected = -1;
let scale = 1, ox = 0, oy = 0;
let img = null, imgF = null;
let imgW = 1, imgH = 1;
let dual = null;
let panning = false, panLast = null;
let drag = null;
let activeFilters = [];
let selFilter = null;
const DUAL_GAP = 8;
const EDGE = 8;
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");

function api(url, opt) {
  return fetch(url, opt).then((r) => {
    if (!r.ok) return r.json().then((j) => { throw new Error(j.error || r.statusText); });
    return r.json();
  });
}
function post(url, body) {
  return api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
}
function showModal(text, extraHtml) {
  document.getElementById("modalText").textContent = text || "";
  document.getElementById("modalExtra").innerHTML = extraHtml || "";
  document.getElementById("modal").classList.remove("hidden");
}
function hideModal() { document.getElementById("modal").classList.add("hidden"); }
document.getElementById("modalOk").onclick = hideModal;
function showLoading(text, pct) {
  document.getElementById("loadingText").textContent = text || "Loading…";
  const bar = document.getElementById("loadingBar");
  const fill = document.getElementById("loadingBarFill");
  const pctEl = document.getElementById("loadingPct");
  if (pct == null || pct === "" || isNaN(+pct)) {
    bar.classList.add("indeterminate");
    fill.style.width = "40%";
    pctEl.textContent = "";
  } else {
    bar.classList.remove("indeterminate");
    const p = Math.max(0, Math.min(100, +pct));
    fill.style.width = p + "%";
    pctEl.textContent = Math.round(p) + "%";
  }
  document.getElementById("loadingOverlay").classList.remove("hidden");
}
function hideLoading() {
  document.getElementById("loadingOverlay").classList.add("hidden");
}
let indexPollTimer = null;
function stopIndexPoll() {
  if (indexPollTimer) { clearTimeout(indexPollTimer); indexPollTimer = null; }
}
function pollDatasetIndex() {
  stopIndexPoll();
  showLoading("Scanning dataset labels and sizes…", 0);
  const tick = () => {
    api("/api/index/status").then((j) => {
      if (j.index_error) {
        hideLoading();
        showModal("Dataset index failed:\n" + j.index_error);
        return;
      }
      showLoading(j.index_msg || "Scanning…", j.pct);
      if (j.indexing) {
        indexPollTimer = setTimeout(tick, 200);
        return;
      }
      if (j.index_ready) {
        api("/api/state").then((st) => {
          applyState(st);
          return api("/api/stats");
        }).then((j2) => {
          hideLoading();
          let t = "Scan finished\n\n" + (j2.text || "");
          if (pendingTrackInfo && pendingTrackInfo.message) t += "\n\n" + pendingTrackInfo.message;
          pendingTrackInfo = null;
          showModal(t);
        }).catch((err) => {
          hideLoading();
          pendingTrackInfo = null;
          showModal("Scan finished\n\n" + (err.message || String(err)));
        });
        return;
      }
      post("/api/index/build")
        .then(() => { indexPollTimer = setTimeout(tick, 200); })
        .catch((err) => { hideLoading(); showModal(err.message || String(err)); });
    }).catch((err) => { hideLoading(); showModal(err.message || String(err)); });
  };
  tick();
}
let pendingTrackInfo = null;
function afterLabelsFolder(st) {
  pendingTrackInfo = st.track_info || null;
  if (st.images_dir && st.labels_dir) pollDatasetIndex();
  else {
    hideLoading();
    if (st.track_info) showModal(st.track_info.message);
  }
}
function maybeStartIndex(st) {
  if (!st || !st.images_dir || !st.labels_dir) return;
  if (st.index_ready && !st.indexing) return;
  pollDatasetIndex();
}

function applyState(st, keepView) {
  S = st;
  document.getElementById("boxTh").value = st.box_thickness;
  document.getElementById("labSz").value = st.label_size;
  document.getElementById("placeW").value = st.place_w;
  document.getElementById("placeH").value = st.place_h;
  document.getElementById("trackChk").checked = st.tracking;
  document.getElementById("checkedBtn").classList.toggle("hidden", !st.tracking);
  fillSelect(document.getElementById("newCls"), st.classes, null);
  fillClassFilter(st);
  fillClusterJump(st);
  const sortMode = st.sort_mode || (st.sort_by_cluster ? "cluster" : "name");
  document.querySelectorAll('input[name="sort"]').forEach((r) => { r.checked = r.value === sortMode; });
  const sizeOn = !!st.size_filter_on;
  const enableSize = document.getElementById("enableSizeFilter");
  if (enableSize) enableSize.checked = sizeOn;
  if (document.getElementById("sizeMin")) {
    const lo = st.size_filter_lo != null ? st.size_filter_lo : 0;
    const hi = st.size_filter_hi != null ? st.size_filter_hi : 1;
    document.getElementById("sizeMin").value = lo;
    document.getElementById("sizeMax").value = hi;
    document.getElementById("sizeMinVal").textContent = Number(lo).toFixed(2);
    document.getElementById("sizeMaxVal").textContent = Number(hi).toFixed(2);
  }
  syncFilterBarForPage();
  renderNav(st);
  document.getElementById("statusLine").textContent =
    (st.index + 1) + "/" + st.total + "  |  " + st.name + "  |  boxes=" + st.boxes.length +
    "  |  mode=" + mode +
    (st.clustering ? "  |  " + st.cluster_msg : (st.dataset_clustered ? "  |  clusters=" + st.cluster_count : ""));
  document.getElementById("statusImg").textContent = "images: " + (st.images_dir || "");
  document.getElementById("statusLab").textContent = "labels: " + (st.labels_dir || "");
  const di = document.getElementById("dirImages");
  const dl = document.getElementById("dirLabels");
  if (di && document.activeElement !== di) di.value = st.images_dir || "";
  if (dl && document.activeElement !== dl) dl.value = st.labels_dir || "";
  updateMultiBoxWarn();
  loadImages(keepView);
}

function updateMultiBoxWarn() {
  const el = document.getElementById("multiBoxWarn");
  const crop = document.getElementById("cropFocus");
  if (!el || !crop) return;
  const n = (S && S.boxes) ? S.boxes.length : 0;
  const on = crop.checked && n > 1;
  el.classList.toggle("hidden", !on);
  el.textContent = on ? (n + " Box Present !") : "";
}

function fillSelect(el, names, selectedIdx) {
  const cur = selectedIdx == null ? el.selectedIndex : selectedIdx;
  el.innerHTML = names.map((n, i) => "<option value='" + i + "'>" + n + "</option>").join("");
  el.selectedIndex = Math.max(0, Math.min(names.length - 1, cur));
}
function fillClassFilter(st) {
  const el = document.getElementById("classFilter");
  let html = "<option value=''>Show All</option>";
  st.classes.forEach((n, i) => { html += "<option value='" + i + "'>Show Class " + i + ": " + n + "</option>"; });
  el.innerHTML = html;
  el.value = st.class_filter == null ? "" : String(st.class_filter);
}
function fillClusterJump(st) {
  const el = document.getElementById("clusterJump");
  const jump = document.getElementById("jumpRow");
  const sortMode = st.sort_mode || (st.sort_by_cluster ? "cluster" : "name");
  const use = sortMode === "cluster" && st.dataset_clustered;
  el.classList.toggle("hidden", !use);
  jump.classList.toggle("hidden", use);
  const opts = (st.clusters || []).map((c) => "<option value='" + c.i + "'>C" + (c.i + 1) + " (" + c.n + ")</option>").join("");
  el.innerHTML = opts;
  const cropSel = document.getElementById("cropClusterJump");
  if (cropSel) {
    const prev = cropSel.value;
    cropSel.innerHTML = opts;
    if (st.crop_cluster_id != null) cropSel.value = String(st.crop_cluster_id);
    else if (prev !== "" && [...cropSel.options].some((o) => o.value === prev)) cropSel.value = prev;
  }
  syncCropClusterNav(st);
}
function syncCropClusterNav(st) {
  const wrap = document.getElementById("cropClusterNav");
  if (!wrap) return;
  const s = st || S;
  const sortMode = s && (s.sort_mode || (s.sort_by_cluster ? "cluster" : "name"));
  const show = isCropPage() && sortMode === "cluster" && s && s.dataset_clustered;
  wrap.classList.toggle("hidden", !show);
  const en = document.getElementById("enableCropCluster");
  if (en && s) en.checked = s.crop_cluster_id != null;
  const sel = document.getElementById("cropClusterJump");
  const prev = document.getElementById("cropClusterPrev");
  const next = document.getElementById("cropClusterNext");
  const on = !!(en && en.checked);
  if (sel) sel.classList.toggle("hidden", !on);
  if (prev) prev.classList.toggle("hidden", !on);
  if (next) next.classList.toggle("hidden", !on);
}
function cropClusterStep(dir) {
  if (!S || !S.dataset_clustered || !(S.clusters || []).length) return;
  const n = S.clusters.length;
  let ci = S.crop_cluster_id;
  if (ci == null) ci = 0;
  ci = (ci + dir + n) % n;
  post("/api/crop_cluster", { cluster: ci }).then(afterFilter);
}

function renderNav(st) {
  const host = document.getElementById("navList");
  const track = st.tracking;
  host.innerHTML = st.nav.map((it) => {
    const on = it.i === st.index ? " on" : "";
    const box = track ? "<span class='box" + (it.checked ? " ok" : "") + "'></span>" : "";
    return "<div class='nav-item" + on + "' data-i='" + it.i + "'>" + box +
      "<span class='num'>" + (it.i + 1) + "</span><span>" + it.name + "</span></div>";
  }).join("");
  const cur = host.querySelector(".on");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function loadImages(keepView) {
  if (!S || S.index < 0) { img = imgF = null; draw(); return; }
  const src = "/api/image?t=" + Date.now();
  const a = new Image();
  a.onload = () => {
    img = a; imgW = S.img_w; imgH = S.img_h;
    dual = selFilter ? (imgW >= imgH ? "v" : "h") : null;
    // Crop Focus only on navigate (next/prev), not after box edits (keepView)
    if (keepView) draw();
    else if (document.getElementById("cropFocus").checked) focusFirstBox();
    else fit();
  };
  a.src = src;
  if (selFilter) {
    const b = new Image();
    b.onload = () => { imgF = b; draw(); };
    b.src = src + "&filter=" + selFilter;
  } else imgF = null;
}

function viewSize() {
  if (dual === "h") return [imgW * 2 + DUAL_GAP, imgH];
  if (dual === "v") return [imgW, imgH * 2 + DUAL_GAP];
  return [imgW, imgH];
}
function fit() {
  const [vw, vh] = viewSize();
  const r = cv.getBoundingClientRect();
  cv.width = r.width; cv.height = r.height;
  scale = Math.min(r.width / vw, r.height / vh);
  ox = (r.width - vw * scale) / 2;
  oy = (r.height - vh * scale) / 2;
  draw();
}
function focusFirstBox() {
  if (!img || !S || !(S.boxes || []).length) { fit(); return; }
  const [x1, y1, x2, y2] = xyxy(S.boxes[0]);
  // with post-process dual view, include the same box on both images
  let fx1 = x1, fy1 = y1, fx2 = x2, fy2 = y2;
  if (dual === "h") fx2 = x2 + imgW + DUAL_GAP;
  if (dual === "v") fy2 = y2 + imgH + DUAL_GAP;
  const bw = Math.max(8, fx2 - fx1), bh = Math.max(8, fy2 - fy1);
  const level = Math.max(1, Math.min(10, +document.getElementById("cropFocusLevel").value || 5));
  // level 5 ≈ previous default (margin 3.2, max ~3.5× fit)
  const margin = dual ? (1.25 + 0.04 * (11 - level)) : (5.5 - 0.46 * level);
  const minMul = 1.05 + 0.03 * level;
  const maxMul = 1.2 + 0.46 * level;
  const r = cv.getBoundingClientRect();
  cv.width = r.width; cv.height = r.height;
  const [vw, vh] = viewSize();
  const fitS = Math.min(r.width / vw, r.height / vh);
  let s = Math.min(r.width / (bw * margin), r.height / (bh * margin));
  if (dual) scale = Math.min(Math.max(s, fitS), fitS * maxMul);
  else scale = Math.min(Math.max(s, fitS * minMul), fitS * maxMul);
  const cx = (fx1 + fx2) / 2, cy = (fy1 + fy2) / 2;
  ox = r.width / 2 - cx * scale;
  oy = r.height / 2 - cy * scale;
  draw();
}
function toImg(mx, my) {
  let x = (mx - ox) / scale, y = (my - oy) / scale;
  if (dual === "h" && x >= imgW + DUAL_GAP / 2) x -= imgW + DUAL_GAP;
  if (dual === "v" && y >= imgH + DUAL_GAP / 2) y -= imgH + DUAL_GAP;
  return [x, y];
}
function xyxy(b) {
  return [
    (b.cx - b.w / 2) * imgW, (b.cy - b.h / 2) * imgH,
    (b.cx + b.w / 2) * imgW, (b.cy + b.h / 2) * imgH,
  ];
}
function toNorm(x1, y1, x2, y2, cls) {
  if (x1 > x2) { const t = x1; x1 = x2; x2 = t; }
  if (y1 > y2) { const t = y1; y1 = y2; y2 = t; }
  x1 = Math.max(0, Math.min(imgW, x1)); x2 = Math.max(0, Math.min(imgW, x2));
  y1 = Math.max(0, Math.min(imgH, y1)); y2 = Math.max(0, Math.min(imgH, y2));
  const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1);
  return { cls, cx: ((x1 + x2) / 2) / imgW, cy: ((y1 + y2) / 2) / imgH, w: bw / imgW, h: bh / imgH };
}
function hitBox(x, y, m) {
  let best = -1, bestD = 1e18;
  (S.boxes || []).forEach((b, i) => {
    const [x1, y1, x2, y2] = xyxy(b);
    if (x1 - m <= x && x <= x2 + m && y1 - m <= y && y <= y2 + m) {
      const d = (x - (x1 + x2) / 2) ** 2 + (y - (y1 + y2) / 2) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
  });
  return best;
}
function hitEdge(x, y, b) {
  const [x1, y1, x2, y2] = xyxy(b);
  const s = popupOpen() ? pScale : scale;
  const m = EDGE / Math.max(s, 1e-6);
  const l = Math.abs(x - x1) <= m, r = Math.abs(x - x2) <= m;
  const t = Math.abs(y - y1) <= m, bot = Math.abs(y - y2) <= m;
  const ix = x1 - m <= x && x <= x2 + m, iy = y1 - m <= y && y <= y2 + m;
  if (t && l) return "tl"; if (t && r) return "tr"; if (bot && l) return "bl"; if (bot && r) return "br";
  if (l && iy) return "l"; if (r && iy) return "r"; if (t && ix) return "t"; if (bot && ix) return "b";
  if (x1 <= x && x <= x2 && y1 <= y && y <= y2) return "move";
  return null;
}

function draw() {
  const r = cv.getBoundingClientRect();
  if (cv.width !== r.width || cv.height !== r.height) { cv.width = r.width; cv.height = r.height; }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!img || !S) return;
  ctx.setTransform(scale, 0, 0, scale, ox, oy);
  ctx.drawImage(img, 0, 0, imgW, imgH);
  let ox2 = 0, oy2 = 0;
  if (imgF && dual === "h") { ox2 = imgW + DUAL_GAP; ctx.drawImage(imgF, ox2, 0, imgW, imgH); }
  if (imgF && dual === "v") { oy2 = imgH + DUAL_GAP; ctx.drawImage(imgF, 0, oy2, imgW, imgH); }
  const offs = [[0, 0]];
  if (imgF && dual) offs.push([ox2, oy2]);
  const showB = document.getElementById("showBoxes").checked;
  const showL = document.getElementById("showLabels").checked;
  const th = S.box_thickness / scale;
  offs.forEach(([dx, dy]) => {
    (S.boxes || []).forEach((b, i) => {
      const [x1, y1, x2, y2] = xyxy(b);
      if (showB) {
        if (i === selected && mode === "edit") {
          ctx.setLineDash([]);
          ctx.strokeStyle = "#000";
          ctx.lineWidth = 5 / scale;
          ctx.strokeRect(x1 + dx, y1 + dy, x2 - x1, y2 - y1);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2 / scale;
          ctx.strokeRect(x1 + dx, y1 + dy, x2 - x1, y2 - y1);
        } else {
          ctx.strokeStyle = S.class_colors[b.cls] || "#0f0";
          ctx.lineWidth = th;
          ctx.strokeRect(x1 + dx, y1 + dy, x2 - x1, y2 - y1);
        }
      }
      if (showL) {
        ctx.fillStyle = S.class_colors[b.cls] || "#0f0";
        ctx.font = (S.label_size / scale) + "px Arial";
        ctx.fillText(S.classes[b.cls] || String(b.cls), x1 + dx, Math.max(dy, y1 + dy - 4));
      }
    });
  });
}

function saveBoxes(changed) {
  return post("/api/boxes", { boxes: S.boxes, changed: !!changed }).then((st) => {
    if (changed) cropPopupDirty = true;
    applyState(st, true);
    if (popupOpen()) drawCropPopup();
  });
}
function setMode(m) {
  mode = (mode === m) ? "view" : m;
  document.getElementById("editBtn").classList.toggle("on", mode === "edit");
  document.getElementById("removeBtn").classList.toggle("on", mode === "remove");
  document.getElementById("placeBtn").classList.toggle("on", mode === "place");
  const pe = document.getElementById("cropPEditBtn");
  const pr = document.getElementById("cropPRemoveBtn");
  const pp = document.getElementById("cropPPlaceBtn");
  if (pe) pe.classList.toggle("on", mode === "edit");
  if (pr) pr.classList.toggle("on", mode === "remove");
  if (pp) pp.classList.toggle("on", mode === "place");
  selected = -1;
  draw();
  if (popupOpen()) drawCropPopup();
  if (S) {
    const el = document.getElementById("statusLine");
    el.textContent = el.textContent.replace(/mode=\w+/, "mode=" + mode);
  }
}

cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const [vw, vh] = viewSize();
  if (!vw || !vh) return;
  const fitS = Math.min(r.width / vw, r.height / vh);
  const minS = fitS * 0.2;
  const maxS = Math.max(fitS * 50, 40);
  let dy = e.deltaY;
  if (e.deltaMode === 1) dy *= 16;
  if (e.deltaMode === 2) dy *= r.height;
  const f = Math.exp(Math.min(50, Math.max(-50, -dy)) * 0.0018);
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const x = (mx - ox) / scale, y = (my - oy) / scale;
  scale = Math.min(maxS, Math.max(minS, scale * f));
  ox = mx - x * scale;
  oy = my - y * scale;
  draw();
}, { passive: false });

cv.addEventListener("mousedown", (e) => {
  if (!S || S.index < 0) return;
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const [x, y] = toImg(mx, my);
  if (e.button === 1 || (e.button === 0 && (mode === "view" || e.ctrlKey))) {
    panning = true; panLast = [mx, my]; return;
  }
  if (e.button !== 0) return;
  if (mode === "place") {
    S.boxes.push(toNorm(x - S.place_w / 2, y - S.place_h / 2, x + S.place_w / 2, y + S.place_h / 2, +document.getElementById("newCls").value));
    selected = S.boxes.length - 1;
    saveBoxes(true); setMode("edit"); return;
  }
  if (mode === "remove") {
    const i = hitBox(x, y, EDGE * 2);
    if (i >= 0) { S.boxes.splice(i, 1); saveBoxes(true); }
    setMode("view"); return;
  }
  if (mode === "edit") {
    const i = hitBox(x, y, EDGE);
    if (i < 0) { panning = true; panLast = [mx, my]; selected = -1; draw(); return; }
    selected = i;
    const edge = hitEdge(x, y, S.boxes[i]) || "move";
    drag = { i, edge, x, y, start: Object.assign({}, S.boxes[i]) };
    draw();
  }
});
cv.addEventListener("mousemove", (e) => {
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  if (panning && panLast) {
    ox += mx - panLast[0]; oy += my - panLast[1]; panLast = [mx, my]; draw(); return;
  }
  if (drag) {
    const [x, y] = toImg(mx, my);
    let [x1, y1, x2, y2] = xyxy(drag.start);
    const dx = x - drag.x, dy = y - drag.y;
    if (drag.edge === "move") { x1 += dx; x2 += dx; y1 += dy; y2 += dy; }
    else {
      if (drag.edge.indexOf("l") >= 0) x1 += dx;
      if (drag.edge.indexOf("r") >= 0) x2 += dx;
      if (drag.edge.indexOf("t") >= 0) y1 += dy;
      if (drag.edge.indexOf("b") >= 0) y2 += dy;
    }
    S.boxes[drag.i] = toNorm(x1, y1, x2, y2, drag.start.cls);
    draw();
  }
});
window.addEventListener("mouseup", () => {
  if (panning) { panning = false; panLast = null; return; }
  if (drag) {
    const changed = JSON.stringify(drag.start) !== JSON.stringify(S.boxes[drag.i]);
    drag = null;
    saveBoxes(changed);
  }
});
cv.addEventListener("dblclick", (e) => {
  if (!S || mode === "place") return;
  const r = cv.getBoundingClientRect();
  const [x, y] = toImg(e.clientX - r.left, e.clientY - r.top);
  const i = hitBox(x, y, EDGE);
  if (i < 0) return;
  selected = i;
  const btns = S.classes.map((n, c) =>
    "<button data-c='" + c + "' style='color:" + S.class_colors[c] + "'>" + c + "  " + n + "</button>"
  ).join("");
  showModal("Select class", "<div class='cls-btns'>" + btns + "</div>");
  document.querySelectorAll(".cls-btns button").forEach((b) => {
    b.onclick = () => {
      const c = +b.dataset.c;
      if (S.boxes[selected].cls !== c) { S.boxes[selected].cls = c; saveBoxes(true); }
      hideModal();
    };
  });
});

document.getElementById("navList").onclick = (e) => {
  const row = e.target.closest(".nav-item");
  if (!row) return;
  const i = +row.dataset.i;
  const jump = document.getElementById("jumpEdit");
  if (jump) jump.value = String(i + 1);
  post("/api/goto", { index: i }).then(applyState);
};
const navToggle = document.getElementById("navToggle");
if (navToggle) {
  navToggle.onclick = () => {
    const nav = document.getElementById("navPanel");
    if (!nav) return;
    const collapsed = nav.classList.toggle("collapsed");
    nav.style.width = "";
    navToggle.innerHTML = collapsed ? "&#9654;" : "&#9664;";
    navToggle.title = collapsed ? "Expand" : "Collapse";
    if (img) fit();
  };
}
document.getElementById("goBtn").onclick = () => post("/api/goto", { index: (+document.getElementById("jumpEdit").value || 1) - 1 }).then(applyState);
document.getElementById("prevBtn").onclick = () => {
  if (!S || S.index <= 0) return;
  post("/api/goto", { index: S.index - 1 }).then(applyState);
};
document.getElementById("nextBtn").onclick = () => {
  if (!S || S.index >= S.total - 1) return;
  post("/api/goto", { index: S.index + 1 }).then(applyState);
};
document.getElementById("jumpEdit").addEventListener("keydown", (e) => { if (e.key === "Enter") document.getElementById("goBtn").click(); });
document.getElementById("clusterJump").onchange = () => post("/api/goto", { cluster: +document.getElementById("clusterJump").value }).then(applyState);
function isCropPage() {
  return document.getElementById("pageCrop").classList.contains("on");
}
function syncFilterBarForPage() {
  const crop = isCropPage();
  const classOn = crop || document.getElementById("enableClassFilter").checked;
  const clusterOn = crop || document.getElementById("enableClusterSort").checked;
  const sizeOn = crop || document.getElementById("enableSizeFilter").checked;
  document.getElementById("classFilterWrap").classList.toggle("hidden", !classOn);
  document.getElementById("clusterSortWrap").classList.toggle("hidden", !clusterOn);
  document.getElementById("sizeFilterWrap").classList.toggle("hidden", !sizeOn);
  syncCropClusterNav(S);
}
let cropShown = 0, cropTotal = 0, cropGen = 0, cropLoadId = 0;
let cropRestoreScroll = null;
let cropFocusHint = null;
let cropPopupDirty = false;
function cropShowCount() {
  const el = document.getElementById("cropShow");
  const v = el ? el.value : "100";
  return v === "all" ? 999999 : (+v || 100);
}
function afterFilter(st, keepView) {
  applyState(st, keepView !== false);
  if (isCropPage() && !(st && st.clustering)) resetAndLoadCrops(cropShowCount());
}
function resetAndLoadCrops(n, opts) {
  opts = opts || {};
  cropRestoreScroll = opts.keepScroll != null ? opts.keepScroll : null;
  if (opts.focusHint) cropFocusHint = opts.focusHint;
  cropShown = 0;
  cropLoadId += 1;
  document.getElementById("cropGrid").innerHTML = "";
  document.getElementById("cropStatus").textContent = "Loading crops…";
  if (!opts.quiet) showLoading("Loading crops…\nPlease wait.");
  loadCrops(n == null ? cropShowCount() : n, cropLoadId);
}

document.getElementById("classFilter").onchange = () => {
  const v = document.getElementById("classFilter").value;
  document.getElementById("classFilter").blur();
  post("/api/filter_class", { cls: v === "" ? null : +v }).then(afterFilter);
};
document.getElementById("enableClassFilter").onchange = () => {
  const on = document.getElementById("enableClassFilter").checked;
  syncFilterBarForPage();
  if (!on) {
    document.getElementById("classFilter").value = "";
    post("/api/filter_class", { cls: null }).then(afterFilter);
  }
};
document.getElementById("enableClusterSort").onchange = () => {
  const on = document.getElementById("enableClusterSort").checked;
  syncFilterBarForPage();
  if (!on) {
    document.querySelector('input[name="sort"][value="name"]').checked = true;
    post("/api/sort", { mode: "name" }).then(afterFilter).catch(() => {});
  }
};
document.getElementById("enableSizeFilter") && (document.getElementById("enableSizeFilter").onchange = () => {
  const on = document.getElementById("enableSizeFilter").checked;
  if (!on) {
    syncFilterBarForPage();
    post("/api/size_filter", { enabled: false }).then(afterFilter);
    return;
  }
  const finishEnable = (st) => {
    document.getElementById("sizeMin").value = "0";
    document.getElementById("sizeMax").value = "1";
    document.getElementById("sizeMinVal").textContent = "0.00";
    document.getElementById("sizeMaxVal").textContent = "1.00";
    afterFilter(st);
  };
  if (S && S.size_filter_ready) {
    post("/api/size_filter", { enabled: true, lo: 0, hi: 1 }).then(finishEnable)
      .catch((err) => {
        document.getElementById("enableSizeFilter").checked = false;
        syncFilterBarForPage();
        showModal(err.message || String(err));
      });
    return;
  }
  maybeStartIndex(S || {});
  document.getElementById("enableSizeFilter").checked = false;
  syncFilterBarForPage();
  showModal("Dataset index is still building.\nWait for the scan to finish, then enable size filter again.");
});
function applySizeRange() {
  const minEl = document.getElementById("sizeMin");
  const maxEl = document.getElementById("sizeMax");
  if (!minEl || !maxEl) return;
  let lo = +minEl.value;
  let hi = +maxEl.value;
  if (lo > hi) { const t = lo; lo = hi; hi = t; }
  document.getElementById("sizeMinVal").textContent = lo.toFixed(2);
  document.getElementById("sizeMaxVal").textContent = hi.toFixed(2);
  post("/api/size_filter", { enabled: true, lo, hi }).then(afterFilter);
}
if (document.getElementById("sizeMin")) {
  document.getElementById("sizeMin").oninput = () => {
    document.getElementById("sizeMinVal").textContent = (+document.getElementById("sizeMin").value).toFixed(2);
  };
  document.getElementById("sizeMax").oninput = () => {
    document.getElementById("sizeMaxVal").textContent = (+document.getElementById("sizeMax").value).toFixed(2);
  };
  document.getElementById("sizeMin").onchange = applySizeRange;
  document.getElementById("sizeMax").onchange = applySizeRange;
}
document.querySelectorAll('input[name="sort"]').forEach((r) => {
  r.onchange = () => {
    const needBuild = (r.value === "size_asc" || r.value === "size_desc") && !(S && S.size_filter_ready);
    if (needBuild) {
      maybeStartIndex(S || {});
      showModal("Dataset index is still building.\nWait for the scan to finish, then try size sort again.");
      r.checked = false;
      document.querySelector('input[name="sort"][value="name"]').checked = true;
      return;
    }
    post("/api/sort", { mode: r.value })
      .then((st) => afterFilter(st))
      .catch((err) => {
        r.checked = false;
        document.querySelector('input[name="sort"][value="name"]').checked = true;
        showModal(err.message);
      });
  };
});
document.getElementById("clusterBtn").onclick = () => {
  showLoading("Clustering dataset…", 0);
  post("/api/cluster").then((st) => {
    applyState(st, true);
    pollCluster();
  }).catch((err) => {
    hideLoading();
    showModal(err.message || String(err));
  });
};
function pollCluster() {
  const tick = () => {
    api("/api/state").then((st) => {
      applyState(st, true);
      showLoading(st.cluster_msg || "Clustering…", st.cluster_pct);
      if (st.clustering) {
        setTimeout(tick, 250);
        return;
      }
      hideLoading();
      if (st.cluster_msg && st.dataset_clustered) showModal(st.cluster_msg);
      if (isCropPage()) resetAndLoadCrops(cropShowCount());
    }).catch((err) => {
      hideLoading();
      showModal(err.message || String(err));
    });
  };
  tick();
}
document.getElementById("showBoxes").onchange = draw;
document.getElementById("showLabels").onchange = draw;
document.getElementById("cropFocus").onchange = () => {
  updateMultiBoxWarn();
  if (document.getElementById("cropFocus").checked) focusFirstBox();
  else fit();
};
const cropFocusLevel = document.getElementById("cropFocusLevel");
if (cropFocusLevel) {
  cropFocusLevel.oninput = () => {
    const lab = document.getElementById("cropFocusLevelVal");
    if (lab) lab.textContent = cropFocusLevel.value;
    if (document.getElementById("cropFocus").checked) focusFirstBox();
  };
}
document.getElementById("editBtn").onclick = () => setMode("edit");
document.getElementById("removeBtn").onclick = () => setMode("remove");
document.getElementById("placeBtn").onclick = () => setMode("place");
document.getElementById("cropPEditBtn").onclick = () => setMode("edit");
document.getElementById("cropPRemoveBtn").onclick = () => setMode("remove");
document.getElementById("cropPPlaceBtn").onclick = () => setMode("place");
document.getElementById("cropPShowBoxes").onchange = () => { if (popupOpen()) drawCropPopup(); };
document.getElementById("cropPShowLabels").onchange = () => { if (popupOpen()) drawCropPopup(); };
document.getElementById("fitBtn").onclick = fit;
document.getElementById("checkedBtn").onclick = () => post("/api/checked").then(applyState);
let delConfirmOpen = false;
function doDelete() {
  post("/api/delete").then(applyState).catch((err) => showModal(err.message || String(err)));
}
function closeDelConfirm() {
  delConfirmOpen = false;
  document.getElementById("modalOk").classList.remove("hidden");
  hideModal();
}
function openDelConfirm() {
  delConfirmOpen = true;
  showModal(
    "Delete this image and its label file?\n\n" + ((S && S.name) || ""),
    "<div class='modal-actions'><button id='delYes' class='danger'>Delete (Enter)</button>" +
    "<button id='delNo'>Cancel (Esc)</button></div>"
  );
  document.getElementById("modalOk").classList.add("hidden");
  document.getElementById("delYes").onclick = () => { closeDelConfirm(); doDelete(); };
  document.getElementById("delNo").onclick = closeDelConfirm;
}
document.getElementById("delBtn").onclick = () => {
  if (!S || S.index < 0) return;
  if (document.getElementById("delWarnChk").checked) openDelConfirm();
  else doDelete();
};
document.getElementById("statsBtn").onclick = () => api("/api/stats").then((j) => showModal(j.text));
function parentDir(p) {
  if (!p) return "";
  const s = String(p).replace(/[\\/]+$/, "");
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  if (i <= 0) return s;
  if (i === 2 && s[1] === ":") return s.slice(0, 3);
  return s.slice(0, i);
}
let appBusy = false;
function setBusy(on) {
  appBusy = on;
  document.getElementById("busyBlock").classList.toggle("hidden", !on);
}
function applyDir(kind, path) {
  if (appBusy) return;
  setBusy(true);
  showLoading(kind === "images" ? "Loading images folder…" : "Starting dataset scan…");
  post("/api/set_dir", { kind, path })
    .then((st) => {
      setBusy(false);
      applyState(st);
      if (kind === "labels") afterLabelsFolder(st);
      else hideLoading();
    })
    .catch((err) => { setBusy(false); hideLoading(); showModal(err.message || String(err)); });
}
function browseDir(kind, initial) {
  if (appBusy) return;
  setBusy(true);
  post("/api/browse", { kind, initial: initial || undefined })
    .then((j) => {
      setBusy(false);
      if (j.path) applyDir(kind, j.path);
    })
    .catch((err) => { setBusy(false); showModal(err.message || String(err)); });
}
document.getElementById("browseImages").onclick = () => browseDir("images");
document.getElementById("browseLabels").onclick = () =>
  browseDir("labels", parentDir(S && S.images_dir) || (S && S.labels_dir) || "");
document.getElementById("applyImagesDir").onclick = () => applyDir("images", document.getElementById("dirImages").value);
document.getElementById("applyLabelsDir").onclick = () => applyDir("labels", document.getElementById("dirLabels").value);
document.getElementById("dirImages").addEventListener("keydown", (e) => { if (e.key === "Enter") document.getElementById("applyImagesDir").click(); });
document.getElementById("dirLabels").addEventListener("keydown", (e) => { if (e.key === "Enter") document.getElementById("applyLabelsDir").click(); });
function saveSettings() {
  post("/api/settings", {
    box_thickness: +document.getElementById("boxTh").value,
    label_size: +document.getElementById("labSz").value,
    place_w: +document.getElementById("placeW").value,
    place_h: +document.getElementById("placeH").value,
    filter_params: S.filter_params,
  }).then((st) => applyState(st, true));
}
["boxTh", "labSz", "placeW", "placeH"].forEach((id) => document.getElementById(id).onchange = saveSettings);
document.getElementById("trackChk").onchange = () => {
  post("/api/track", { enabled: document.getElementById("trackChk").checked })
    .then((st) => { applyState(st, true); if (st.track_info) showModal(st.track_info.message); })
    .catch((err) => { document.getElementById("trackChk").checked = false; showModal(err.message); });
};
document.getElementById("setClasses").onclick = () => {
  const names = (S.classes || []).join("\n");
  showModal("One class name per line", "<textarea id='clsArea' rows='8'>" + names + "</textarea>");
  document.getElementById("modalOk").onclick = () => {
    const list = document.getElementById("clsArea").value.split(/\r?\n/);
    hideModal();
    document.getElementById("modalOk").onclick = hideModal;
    post("/api/classes", { names: list }).then((st) => applyState(st, true));
  };
};
document.getElementById("darkChk").onchange = () => {
  document.body.classList.toggle("light", !document.getElementById("darkChk").checked);
};

function updateFilterPanel() {
  const panel = document.getElementById("filterPanel");
  const radios = document.getElementById("filterRadios");
  const sliders = document.getElementById("filterSliders");
  if (!activeFilters.length) { panel.classList.add("hidden"); selFilter = null; loadImages(true); return; }
  panel.classList.remove("hidden");
  if (activeFilters.indexOf(selFilter) < 0) selFilter = activeFilters[0];
  radios.innerHTML = activeFilters.map((f) =>
    "<label class='row'><input type='radio' name='fr' value='" + f + "'" + (f === selFilter ? " checked" : "") + " /> " + f + "</label>"
  ).join("");
  radios.querySelectorAll("input").forEach((el) => { el.onchange = () => { selFilter = el.value; updateFilterPanel(); }; });
  const p = S.filter_params[selFilter] || {};
  let html = "";
  if (selFilter === "gamma") html = sliderHtml("value", "Gamma", p.value, 0.1, 3, 0.01);
  if (selFilter === "clahe") html = sliderHtml("clip", "Clip", p.clip, 1, 40, 0.1) + sliderHtml("tile", "Tile", p.tile, 8, 256, 1);
  if (selFilter === "unsharp") html = sliderHtml("amount", "Amount", p.amount, 0.2, 4, 0.01) + sliderHtml("sigma", "Radius", p.sigma, 0.3, 5, 0.1);
  if (selFilter === "emboss") html = sliderHtml("strength", "Strength", p.strength, 0.2, 4, 0.01);
  sliders.innerHTML = html;
  sliders.querySelectorAll("input").forEach((el) => {
    el.oninput = () => {
      S.filter_params[selFilter][el.dataset.k] = +el.value;
      el.nextElementSibling.textContent = (+el.value).toFixed(2);
      saveSettings();
    };
  });
  loadImages(true);
}
function sliderHtml(k, lab, v, a, b, step) {
  return "<label>" + lab + " <input type='range' min='" + a + "' max='" + b + "' step='" + step + "' value='" + v + "' data-k='" + k + "' /> <span>" + Number(v).toFixed(2) + "</span></label>";
}
document.querySelectorAll(".fchk").forEach((el) => {
  el.onchange = () => {
    activeFilters = [...document.querySelectorAll(".fchk")].filter((c) => c.checked).map((c) => c.dataset.f);
    updateFilterPanel();
  };
});

document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("on"));
    t.classList.add("on");
    const page = t.dataset.page;
    document.getElementById("pageViewer").classList.toggle("on", page === "viewer");
    document.getElementById("pageCrop").classList.toggle("on", page === "crop");
    syncFilterBarForPage();
    if (page === "crop") resetAndLoadCrops(cropShowCount());
  };
});
document.getElementById("cropShow").onchange = () => {
  document.getElementById("cropShow").blur();
  resetAndLoadCrops(cropShowCount());
};
document.getElementById("enableCropCluster") && (document.getElementById("enableCropCluster").onchange = () => {
  const on = document.getElementById("enableCropCluster").checked;
  if (!on) {
    post("/api/crop_cluster", { cluster: null }).then(afterFilter);
    return;
  }
  const sel = document.getElementById("cropClusterJump");
  const ci = sel && sel.value !== "" ? +sel.value : (S && S.crop_cluster_id != null ? S.crop_cluster_id : 0);
  post("/api/crop_cluster", { cluster: ci }).then(afterFilter);
});
document.getElementById("cropClusterJump") && (document.getElementById("cropClusterJump").onchange = () => {
  document.getElementById("cropClusterJump").blur();
  if (!document.getElementById("enableCropCluster").checked) return;
  post("/api/crop_cluster", { cluster: +document.getElementById("cropClusterJump").value }).then(afterFilter);
});
document.getElementById("cropClusterPrev") && (document.getElementById("cropClusterPrev").onclick = () => cropClusterStep(-1));
document.getElementById("cropClusterNext") && (document.getElementById("cropClusterNext").onclick = () => cropClusterStep(1));

const pCv = document.getElementById("cropPopupCv");
const pCtx = pCv.getContext("2d");
let pImg = null, pW = 0, pH = 0, pScale = 1, pOx = 0, pOy = 0;
let pPan = false, pLast = null;

function popupOpen() {
  return !document.getElementById("cropPopup").classList.contains("hidden");
}
function pToImg(mx, my) {
  return [(mx - pOx) / pScale, (my - pOy) / pScale];
}
function drawCropPopup() {
  const r = pCv.getBoundingClientRect();
  if (pCv.width !== r.width || pCv.height !== r.height) {
    pCv.width = r.width; pCv.height = r.height;
  }
  pCtx.setTransform(1, 0, 0, 1, 0, 0);
  pCtx.fillStyle = "#0a0a0a";
  pCtx.fillRect(0, 0, pCv.width, pCv.height);
  if (!pImg) return;
  pCtx.setTransform(pScale, 0, 0, pScale, pOx, pOy);
  pCtx.drawImage(pImg, 0, 0, pW, pH);
  if (!S) return;
  const showB = document.getElementById("cropPShowBoxes").checked;
  const showL = document.getElementById("cropPShowLabels").checked;
  const th = (S.box_thickness || 2) / pScale;
  (S.boxes || []).forEach((b, i) => {
    const [x1, y1, x2, y2] = xyxy(b);
    if (showB) {
      if (i === selected && mode === "edit") {
        pCtx.setLineDash([]);
        pCtx.strokeStyle = "#000";
        pCtx.lineWidth = 5 / pScale;
        pCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        pCtx.strokeStyle = "#fff";
        pCtx.lineWidth = 2 / pScale;
        pCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      } else {
        pCtx.strokeStyle = (S.class_colors && S.class_colors[b.cls]) || "#0f0";
        pCtx.lineWidth = th;
        pCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    }
    if (showL) {
      pCtx.fillStyle = (S.class_colors && S.class_colors[b.cls]) || "#0f0";
      pCtx.font = ((S.label_size || 14) / pScale) + "px Arial";
      pCtx.fillText((S.classes && S.classes[b.cls]) || String(b.cls), x1, Math.max(0, y1 - 4));
    }
  });
}
function fitCropPopup() {
  if (!pImg) return;
  const r = pCv.getBoundingClientRect();
  pCv.width = r.width; pCv.height = r.height;
  pScale = Math.min(r.width / pW, r.height / pH);
  pOx = (r.width - pW * pScale) / 2;
  pOy = (r.height - pH * pScale) / 2;
  drawCropPopup();
}
function openCropPopup(it) {
  cropPopupDirty = false;
  cropFocusHint = { name: it.name || "", cls: it.cls, cls_name: it.cls_name || "" };
  post("/api/open_crop", { i: it.i }).then((st) => {
    applyState(st, true);
    selected = -1;
    const sb = st.select_box;
    if (sb && S && S.boxes) {
      selected = S.boxes.findIndex((b) =>
        Math.abs(b.cx - sb.cx) < 1e-5 && Math.abs(b.cy - sb.cy) < 1e-5 &&
        Math.abs(b.w - sb.w) < 1e-5 && Math.abs(b.h - sb.h) < 1e-5
      );
    }
    document.getElementById("cropPopupTitle").textContent =
      (it.name || "") + "  |  " + (it.cls_name || "") + "  |  E edit  R remove  Q place  B boxes  L labels";
    pImg = null;
    document.getElementById("cropPopup").classList.remove("hidden");
    const a = new Image();
    a.onload = () => {
      pImg = a; pW = a.naturalWidth; pH = a.naturalHeight;
      imgW = pW; imgH = pH;
      fitCropPopup();
    };
    a.src = "/api/crop_image/" + it.i + "?g=" + cropGen + "&t=" + Date.now();
  });
}
function closeCropPopup() {
  const was = popupOpen();
  const grid = document.getElementById("cropGrid");
  const scroll = grid ? grid.scrollTop : 0;
  const hint = cropFocusHint;
  document.getElementById("cropPopup").classList.add("hidden");
  pImg = null; pPan = false; pLast = null; drag = null;
  if (was && isCropPage() && cropPopupDirty) {
    resetAndLoadCrops(cropShowCount(), { keepScroll: scroll, focusHint: hint, quiet: true });
  }
  cropPopupDirty = false;
}
document.getElementById("cropPopupClose").onclick = closeCropPopup;
document.getElementById("cropPopup").addEventListener("click", (e) => {
  if (e.target.id === "cropPopup") closeCropPopup();
});
pCv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!pImg) return;
  const r = pCv.getBoundingClientRect();
  const fitS = Math.min(r.width / pW, r.height / pH);
  const minS = fitS * 0.2;
  const maxS = Math.max(fitS * 50, 40);
  let dy = e.deltaY;
  if (e.deltaMode === 1) dy *= 16;
  if (e.deltaMode === 2) dy *= r.height;
  const f = Math.exp(Math.min(50, Math.max(-50, -dy)) * 0.0018);
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const x = (mx - pOx) / pScale, y = (my - pOy) / pScale;
  pScale = Math.min(maxS, Math.max(minS, pScale * f));
  pOx = mx - x * pScale;
  pOy = my - y * pScale;
  drawCropPopup();
}, { passive: false });
pCv.addEventListener("mousedown", (e) => {
  if (!S || !pImg) return;
  const r = pCv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const [x, y] = pToImg(mx, my);
  if (e.button === 1) {
    e.preventDefault();
    const el = document.getElementById("cropPShowBoxes");
    el.checked = !el.checked;
    drawCropPopup();
    return;
  }
  if (e.button === 0 && (mode === "view" || e.ctrlKey)) {
    pPan = true; pLast = [e.clientX, e.clientY];
    pCv.classList.add("dragging");
    return;
  }
  if (e.button !== 0) return;
  imgW = pW; imgH = pH;
  if (mode === "place") {
    S.boxes.push(toNorm(x - S.place_w / 2, y - S.place_h / 2, x + S.place_w / 2, y + S.place_h / 2, +document.getElementById("newCls").value));
    selected = S.boxes.length - 1;
    saveBoxes(true); setMode("edit"); return;
  }
  if (mode === "remove") {
    const i = hitBox(x, y, EDGE * 2);
    if (i >= 0) { S.boxes.splice(i, 1); saveBoxes(true); }
    setMode("view"); return;
  }
  if (mode === "edit") {
    const i = hitBox(x, y, EDGE);
    if (i < 0) {
      pPan = true; pLast = [e.clientX, e.clientY];
      pCv.classList.add("dragging");
      selected = -1; drawCropPopup(); return;
    }
    selected = i;
    const edge = hitEdge(x, y, S.boxes[i]) || "move";
    drag = { i, edge, x, y, start: Object.assign({}, S.boxes[i]), popup: true };
    drawCropPopup();
  }
});
pCv.addEventListener("dblclick", (e) => {
  if (!S || mode === "place" || !pImg) return;
  const r = pCv.getBoundingClientRect();
  imgW = pW; imgH = pH;
  const [x, y] = pToImg(e.clientX - r.left, e.clientY - r.top);
  const i = hitBox(x, y, EDGE);
  if (i < 0) return;
  selected = i;
  const btns = S.classes.map((n, c) =>
    "<button data-c='" + c + "' style='color:" + S.class_colors[c] + "'>" + c + "  " + n + "</button>"
  ).join("");
  showModal("Select class", "<div class='cls-btns'>" + btns + "</div>");
  document.querySelectorAll(".cls-btns button").forEach((b) => {
    b.onclick = () => {
      const c = +b.dataset.c;
      if (S.boxes[selected].cls !== c) { S.boxes[selected].cls = c; saveBoxes(true); }
      hideModal();
    };
  });
});
pCv.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
pCv.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("mousemove", (e) => {
  if (!popupOpen()) return;
  if (pPan && pLast) {
    pOx += e.clientX - pLast[0];
    pOy += e.clientY - pLast[1];
    pLast = [e.clientX, e.clientY];
    drawCropPopup();
    return;
  }
  if (drag && drag.popup) {
    const r = pCv.getBoundingClientRect();
    imgW = pW; imgH = pH;
    const [x, y] = pToImg(e.clientX - r.left, e.clientY - r.top);
    let [x1, y1, x2, y2] = xyxy(drag.start);
    const dx = x - drag.x, dy = y - drag.y;
    if (drag.edge === "move") { x1 += dx; x2 += dx; y1 += dy; y2 += dy; }
    else {
      if (drag.edge.indexOf("l") >= 0) x1 += dx;
      if (drag.edge.indexOf("r") >= 0) x2 += dx;
      if (drag.edge.indexOf("t") >= 0) y1 += dy;
      if (drag.edge.indexOf("b") >= 0) y2 += dy;
    }
    S.boxes[drag.i] = toNorm(x1, y1, x2, y2, drag.start.cls);
    drawCropPopup();
  }
});
window.addEventListener("mouseup", () => {
  pPan = false; pLast = null;
  pCv.classList.remove("dragging");
  if (drag && drag.popup) {
    const changed = JSON.stringify(drag.start) !== JSON.stringify(S.boxes[drag.i]);
    drag = null;
    saveBoxes(changed);
  }
});

function loadCrops(n, loadId) {
  const id = loadId == null ? cropLoadId : loadId;
  api("/api/crops?offset=" + cropShown + "&count=" + n + "&_=" + id).then((j) => {
    if (id !== cropLoadId) return;
    cropGen = j.gen || cropGen;
    cropTotal = j.total;
    const grid = document.getElementById("cropGrid");
    j.items.forEach((it) => {
      const d = document.createElement("div");
      d.className = "crop-cell";
      d.dataset.name = it.name || "";
      d.dataset.cls = String(it.cls);
      d.title = "#" + (it.i + 1) + " | " + it.name + " | " + it.cls_name;
      d.innerHTML = "<img src='/api/crop_thumb/" + it.i + "?g=" + cropGen + "' alt='' loading='lazy' />";
      d.onclick = () => openCropPopup(it);
      grid.appendChild(d);
    });
    cropShown += j.items.length;
    document.getElementById("cropStatus").textContent = cropShown + " / " + cropTotal + " crops";
    hideLoading();
    if (cropFocusHint) {
      const hint = cropFocusHint;
      cropFocusHint = null;
      const hit = [...grid.querySelectorAll(".crop-cell")].find((c) =>
        c.dataset.name === hint.name && (hint.cls == null || String(hint.cls) === c.dataset.cls)
      );
      if (hit) hit.scrollIntoView({ block: "center" });
      else if (cropRestoreScroll != null) grid.scrollTop = cropRestoreScroll;
      cropRestoreScroll = null;
    } else if (cropRestoreScroll != null) {
      grid.scrollTop = cropRestoreScroll;
      cropRestoreScroll = null;
    }
  }).catch((err) => {
    if (id !== cropLoadId) return;
    hideLoading();
    cropRestoreScroll = null;
    cropFocusHint = null;
    document.getElementById("cropStatus").textContent = "Crop load failed";
    showModal("Crop load failed: " + (err.message || err));
  });
}

function isTypingTarget(el) {
  if (!el || el === document.body) return false;
  const tag = el.tagName;
  if (el.isContentEditable) return true;
  if (tag === "TEXTAREA") return true;
  if (tag === "SELECT") return true;
  if (tag === "INPUT") {
    const t = (el.type || "text").toLowerCase();
    return t === "text" || t === "number" || t === "search" || t === "password" || t === "email" || t === "";
  }
  return false;
}

window.addEventListener("keydown", (e) => {
  if (appBusy || !document.getElementById("loadingOverlay").classList.contains("hidden")) {
    e.preventDefault();
    return;
  }
  if (delConfirmOpen) {
    if (e.key === "Enter") { e.preventDefault(); closeDelConfirm(); doDelete(); }
    else if (e.key === "Escape") { e.preventDefault(); closeDelConfirm(); }
    return;
  }
  if (e.key === "Escape") {
    if (!document.getElementById("modal").classList.contains("hidden")) { hideModal(); return; }
    if (popupOpen()) { closeCropPopup(); return; }
    hideModal();
    return;
  }
  if (isTypingTarget(e.target)) return;
  const k = e.key.toLowerCase();
  if (k === "e") setMode("edit");
  if (k === "r") setMode("remove");
  if (k === "q") setMode("place");
  if (k === "f") { if (popupOpen()) fitCropPopup(); else fit(); }
  if (k === "c") {
    if (popupOpen()) return;
    const el = document.getElementById("cropFocus");
    el.checked = !el.checked;
    el.dispatchEvent(new Event("change"));
  }
  if (k === "tab" && S && S.tracking) {
    e.preventDefault();
    post("/api/checked").then(applyState);
  }
  if (k === "b") {
    const el = popupOpen() ? document.getElementById("cropPShowBoxes") : document.getElementById("showBoxes");
    el.checked = !el.checked;
    if (popupOpen()) drawCropPopup(); else draw();
  }
  if (k === "l") {
    const el = popupOpen() ? document.getElementById("cropPShowLabels") : document.getElementById("showLabels");
    el.checked = !el.checked;
    if (popupOpen()) drawCropPopup(); else draw();
  }
  if ((e.ctrlKey || e.metaKey) && k === "z") {
    e.preventDefault();
    post("/api/undo").then((st) => { applyState(st, true); if (popupOpen()) drawCropPopup(); });
  }
  if (k === "arrowleft" || k === "a") {
    if (popupOpen()) return;
    if (isCropPage()) {
      const sortMode = S && (S.sort_mode || (S.sort_by_cluster ? "cluster" : "name"));
      if (sortMode === "cluster" && S.dataset_clustered && document.getElementById("enableCropCluster") && document.getElementById("enableCropCluster").checked) {
        e.preventDefault();
        cropClusterStep(-1);
      }
      return;
    }
    e.preventDefault(); document.getElementById("prevBtn").click();
  }
  if (k === "arrowright" || k === "d") {
    if (popupOpen()) return;
    if (isCropPage()) {
      const sortMode = S && (S.sort_mode || (S.sort_by_cluster ? "cluster" : "name"));
      if (sortMode === "cluster" && S.dataset_clustered && document.getElementById("enableCropCluster") && document.getElementById("enableCropCluster").checked) {
        e.preventDefault();
        cropClusterStep(1);
      }
      return;
    }
    e.preventDefault(); document.getElementById("nextBtn").click();
  }
  if (k === "delete") {
    if (popupOpen()) return;
    e.preventDefault(); document.getElementById("delBtn").click();
  }
});

document.addEventListener("mouseup", (e) => {
  const t = e.target;
  if (!t || isTypingTarget(t)) return;
  if (t.tagName === "BUTTON" || t.tagName === "SUMMARY" || t.closest("button") || t.closest("summary")) {
    const focusEl = document.activeElement;
    if (focusEl && !isTypingTarget(focusEl) && focusEl.tagName !== "SELECT") focusEl.blur();
    return;
  }
  if (t.tagName === "INPUT") {
    const ty = (t.type || "").toLowerCase();
    if (ty === "checkbox" || ty === "radio" || ty === "range") t.blur();
  }
}, true);
window.addEventListener("resize", () => { if (img) fit(); });

(function setupMenus() {
  const menusRoot = document.querySelector(".menus");
  if (!menusRoot) return;
  let closeTimer = null;
  function closeMenus(except) {
    menusRoot.querySelectorAll("details[open]").forEach((d) => {
      if (d !== except) d.removeAttribute("open");
    });
  }
  function cancelClose() {
    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
  }
  menusRoot.querySelectorAll("details").forEach((d) => {
    d.addEventListener("toggle", () => {
      if (d.open) closeMenus(d);
    });
    d.addEventListener("mouseenter", cancelClose);
    d.addEventListener("mouseleave", () => {
      cancelClose();
      closeTimer = setTimeout(() => {
        d.removeAttribute("open");
        closeTimer = null;
      }, 250);
    });
  });
  document.addEventListener("click", (e) => {
    if (!menusRoot.contains(e.target)) closeMenus(null);
  });
  document.getElementById("browseImages").addEventListener("click", () => closeMenus(null));
  document.getElementById("browseLabels").addEventListener("click", () => closeMenus(null));
  document.getElementById("applyImagesDir").addEventListener("click", () => closeMenus(null));
  document.getElementById("applyLabelsDir").addEventListener("click", () => closeMenus(null));
  document.getElementById("setClasses").addEventListener("click", () => closeMenus(null));
})();

(function setupNavSplit() {
  const nav = document.querySelector(".nav");
  const split = document.getElementById("navSplit");
  if (!nav || !split) return;
  let dragging = false;
  split.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    document.body.classList.add("nav-resizing");
    split.classList.add("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const page = document.getElementById("pageViewer");
    const left = page.getBoundingClientRect().left;
    const w = Math.max(160, Math.min(window.innerWidth * 0.6, e.clientX - left));
    nav.style.width = w + "px";
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("nav-resizing");
    split.classList.remove("dragging");
    if (img) fit();
  });
})();

setTimeout(function () {
  api("/api/state").then((st) => {
    applyState(st);
    maybeStartIndex(st);
    activeFilters = [...document.querySelectorAll(".fchk")].filter((c) => c.checked).map((c) => c.dataset.f);
    if (activeFilters.length) updateFilterPanel();
  }).catch((err) => {
    console.error(err);
    showModal("Failed to load state: " + (err && err.message ? err.message : err));
  });
}, 0);
