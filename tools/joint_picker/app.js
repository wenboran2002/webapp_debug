/*
  4DHOI Joint Picker
  - Loads an image and a joints JSON { name: [x, y], ... }
  - Select a joint then click to set coordinates
  - Drag to fine-tune
  - Export JSON
*/

const DEFAULT_IMAGE = "../../asset/data/human_kp.png";
const DEFAULT_JOINTS = "../../asset/data/main_joint.json";
const DEFAULT_NAMES = "../../asset/data/button_name.json";

/** @typedef {{[name: string]: [number, number]}} Joints2D */

const els = {
  canvas: document.getElementById("canvas"),
  wrap: document.getElementById("canvas-wrap"),

  btnLoadDefault: document.getElementById("btn-load-default"),
  btnLoadMain: document.getElementById("btn-load-main"),
  btnLoadNames: document.getElementById("btn-load-names"),
  fileImage: document.getElementById("file-image"),
  fileJoints: document.getElementById("file-joints"),

  zoom: document.getElementById("zoom"),
  zoomLabel: document.getElementById("zoom-label"),
  radius: document.getElementById("radius"),
  radiusLabel: document.getElementById("radius-label"),

  search: document.getElementById("search"),
  jointSelect: document.getElementById("joint-select"),
  jointList: document.getElementById("joint-list"),

  coordX: document.getElementById("coord-x"),
  coordY: document.getElementById("coord-y"),
  btnSetCoord: document.getElementById("btn-set-coord"),

  newName: document.getElementById("new-name"),
  btnAdd: document.getElementById("btn-add"),
  btnUndo: document.getElementById("btn-undo"),
  btnReset: document.getElementById("btn-reset"),

  btnDownload: document.getElementById("btn-download"),
  btnCopy: document.getElementById("btn-copy"),

  status: document.getElementById("status"),
  cursor: document.getElementById("cursor"),
  selected: document.getElementById("selected"),
};

const ctx = els.canvas.getContext("2d");

/** @type {HTMLImageElement} */
let image = new Image();
image.crossOrigin = "anonymous";

/** @type {Joints2D} */
let joints = {};
/** @type {Joints2D|null} */
let baselineJoints = null;

let selectedJoint = null;
let zoom = 1.0;
let pointRadius = 7;

let dragging = false;
let dragJointName = null;
let dragOffset = [0, 0];
let dragPrevPoint = null;

/** @type {Array<{type: 'set'|'add'|'delete', name: string, prev?: [number,number], next?: [number,number]}>} */
let undoStack = [];

function setStatus(msg, kind = "") {
  const prefix = kind === "ok" ? "[OK] " : kind === "err" ? "[ERR] " : "";
  els.status.textContent = prefix + msg;
  els.status.style.color = kind === "ok" ? "var(--ok)" : kind === "err" ? "var(--danger)" : "var(--muted)";
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function nowKey() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function getJointNames() {
  return Object.keys(joints).sort((a, b) => a.localeCompare(b));
}

function setSelectedJoint(name) {
  selectedJoint = name;
  els.selected.textContent = `selected: ${name ?? "-"}`;

  if (name && joints[name] && isFinitePoint(joints[name])) {
    const [x, y] = joints[name];
    els.coordX.value = String(Math.round(x));
    els.coordY.value = String(Math.round(y));
  } else {
    els.coordX.value = "";
    els.coordY.value = "";
  }

  // sync select
  if (name) {
    els.jointSelect.value = name;
  }

  renderJointList();
  draw();
}

function renderJointSelect() {
  const names = getJointNames();
  els.jointSelect.innerHTML = "";
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.jointSelect.appendChild(opt);
  }
  if (selectedJoint && names.includes(selectedJoint)) {
    els.jointSelect.value = selectedJoint;
  } else {
    selectedJoint = names[0] ?? null;
    if (selectedJoint) els.jointSelect.value = selectedJoint;
  }
  setSelectedJoint(selectedJoint);
}

function renderJointList() {
  const q = (els.search.value || "").trim().toLowerCase();
  const names = getJointNames().filter((n) => (q ? n.toLowerCase().includes(q) : true));

  els.jointList.innerHTML = "";
  for (const name of names) {
    const item = document.createElement("div");
    item.className = "list-item" + (name === selectedJoint ? " active" : "");

    const left = document.createElement("div");
    left.textContent = name;

    const right = document.createElement("div");
    right.className = "badge";
    const [x, y] = joints[name] ?? [NaN, NaN];
    right.textContent = Number.isFinite(x) ? `${Math.round(x)}, ${Math.round(y)}` : "-";

    item.appendChild(left);
    item.appendChild(right);

    item.addEventListener("click", () => setSelectedJoint(name));
    els.jointList.appendChild(item);
  }
}

function setZoom(z) {
  zoom = clamp(z, 0.25, 3);
  els.zoom.value = String(zoom);
  els.zoomLabel.textContent = `${zoom.toFixed(2)}×`;

  // Keep scroll position roughly centered
  const prevCenterX = (els.wrap.scrollLeft + els.wrap.clientWidth / 2) / (els.canvas.width || 1);
  const prevCenterY = (els.wrap.scrollTop + els.wrap.clientHeight / 2) / (els.canvas.height || 1);

  resizeCanvasToImage();

  els.wrap.scrollLeft = prevCenterX * els.canvas.width - els.wrap.clientWidth / 2;
  els.wrap.scrollTop = prevCenterY * els.canvas.height - els.wrap.clientHeight / 2;
}

function setRadius(r) {
  pointRadius = clamp(r, 2, 40);
  els.radius.value = String(pointRadius);
  els.radiusLabel.textContent = String(pointRadius);
  draw();
}

function resizeCanvasToImage() {
  const w = Math.max(1, Math.floor(image.naturalWidth * zoom));
  const h = Math.max(1, Math.floor(image.naturalHeight * zoom));
  els.canvas.width = w;
  els.canvas.height = h;
  draw();
}

function imageToCanvas(xImg, yImg) {
  return [xImg * zoom, yImg * zoom];
}

function canvasToImage(xCanvas, yCanvas) {
  return [xCanvas / zoom, yCanvas / zoom];
}

function draw() {
  if (!image || !image.complete || image.naturalWidth === 0) {
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    return;
  }

  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(image, 0, 0, els.canvas.width, els.canvas.height);

  // points
  const names = getJointNames();
  for (const name of names) {
    const p = joints[name];
    if (!p) continue;
    const [x, y] = p;
    const [cx, cy] = imageToCanvas(x, y);

    const isSelected = name === selectedJoint;
    ctx.beginPath();
    ctx.arc(cx, cy, pointRadius, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? "rgba(96,165,250,0.85)" : "rgba(230,232,242,0.75)";
    ctx.fill();

    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.strokeStyle = isSelected ? "rgba(96,165,250,1.0)" : "rgba(0,0,0,0.65)";
    ctx.stroke();

    // label
    ctx.font = `${Math.max(10, Math.round(12 * zoom))}px ui-monospace, monospace`;
    ctx.fillStyle = isSelected ? "rgba(96,165,250,1.0)" : "rgba(0,0,0,0.75)";
    ctx.fillText(name, cx + pointRadius + 4, cy - pointRadius - 2);
  }
}

function getMousePosCanvas(evt) {
  const rect = els.canvas.getBoundingClientRect();
  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  return [x, y];
}

function findJointNearCanvas(xCanvas, yCanvas, radiusPx) {
  const names = getJointNames();
  for (const name of names) {
    const p = joints[name];
    if (!p) continue;
    const [cx, cy] = imageToCanvas(p[0], p[1]);
    const dx = xCanvas - cx;
    const dy = yCanvas - cy;
    if (dx * dx + dy * dy <= radiusPx * radiusPx) return name;
  }
  return null;
}

function pushUndo(entry) {
  undoStack.push(entry);
  if (undoStack.length > 200) undoStack.shift();
}

function setJointPoint(name, xImg, yImg) {
  const prev = joints[name] ? [...joints[name]] : undefined;
  const next = [xImg, yImg];
  joints[name] = next;
  pushUndo({ type: "set", name, prev, next });
  setSelectedJoint(name);
}

function addJoint(name) {
  if (!name) return;
  if (joints[name]) {
    setStatus(`已存在：${name}`, "err");
    return;
  }
  joints[name] = [Math.round(image.naturalWidth / 2), Math.round(image.naturalHeight / 2)];
  pushUndo({ type: "add", name, next: [...joints[name]] });
  renderJointSelect();
  setSelectedJoint(name);
  setStatus(`已添加：${name}（默认放在中心点）`, "ok");
}

function resetJoints() {
  if (!baselineJoints) {
    setStatus("没有可重置的 baseline（请先加载一个 joints JSON）", "err");
    return;
  }
  joints = JSON.parse(JSON.stringify(baselineJoints));
  undoStack = [];
  renderJointSelect();
  renderJointList();
  draw();
  setStatus("已重置到加载时的状态", "ok");
}

function undo() {
  const last = undoStack.pop();
  if (!last) {
    setStatus("没有可撤销的操作", "");
    return;
  }

  if (last.type === "set") {
    if (last.prev) joints[last.name] = last.prev;
    else delete joints[last.name];
    setSelectedJoint(last.name);
  } else if (last.type === "add") {
    delete joints[last.name];
    if (selectedJoint === last.name) selectedJoint = null;
    renderJointSelect();
  } else if (last.type === "delete") {
    joints[last.name] = last.prev;
    renderJointSelect();
    setSelectedJoint(last.name);
  }

  setStatus("已撤销", "ok");
  renderJointList();
  draw();
}

function downloadJSON() {
  const payload = JSON.stringify(joints, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `main_joint_${nowKey()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus("已下载 JSON", "ok");
}

async function copyJSON() {
  const payload = JSON.stringify(joints, null, 2);
  try {
    await navigator.clipboard.writeText(payload);
    setStatus("已复制 JSON 到剪贴板", "ok");
  } catch {
    setStatus("复制失败（浏览器权限限制），建议使用下载 JSON", "err");
  }
}

async function loadImageFromUrl(url) {
  setStatus(`加载图片：${url} ...`);
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("failed to load image"));
    img.src = url;
  });
  image = img;
  resizeCanvasToImage();
  setStatus(`图片已加载：${image.naturalWidth}×${image.naturalHeight}`, "ok");
}

async function loadJSONFromUrl(url) {
  setStatus(`加载 JSON：${url} ...`);
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function normalizeJoints2D(obj) {
  // expects {name: [x,y], ...}
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    if (!Array.isArray(v) || v.length < 2) continue;
    const x = Number(v[0]);
    const y = Number(v[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    out[String(k)] = [x, y];
  }
  return out;
}

async function loadJointsFromUrl(url) {
  const obj = await loadJSONFromUrl(url);
  joints = normalizeJoints2D(obj);
  baselineJoints = JSON.parse(JSON.stringify(joints));
  undoStack = [];
  renderJointSelect();
  renderJointList();
  draw();
  setStatus(`关节已加载：${Object.keys(joints).length} 个`, "ok");
}

async function loadNamesFromUrl(url) {
  const obj = await loadJSONFromUrl(url);
  // button_name.json is {name: shortName}
  const names = Object.keys(obj || {});
  // keep existing points if any
  const merged = { ...joints };
  for (const name of names) {
    if (!merged[name]) merged[name] = [NaN, NaN];
  }
  // Remove NaNs when drawing/listing? We'll keep NaN but hide in draw()
  // So convert NaNs to undefined by leaving entry but checking finite before drawing
  // Normalize to [x,y] only if finite; else delete coords for that joint.
  const cleaned = {};
  for (const [k, v] of Object.entries(merged)) {
    if (!Array.isArray(v) || v.length < 2) continue;
    const x = Number(v[0]);
    const y = Number(v[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) cleaned[k] = [x, y];
    else cleaned[k] = [NaN, NaN];
  }
  joints = cleaned;
  baselineJoints = JSON.parse(JSON.stringify(joints));
  undoStack = [];
  renderJointSelect();
  renderJointList();
  draw();
  setStatus(`已导入关节名：${names.length} 个（未设置坐标的显示为 -）`, "ok");
}

function isFinitePoint(p) {
  return Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
}

// override draw to ignore NaN points
const _draw = draw;
draw = function () {
  if (!image || !image.complete || image.naturalWidth === 0) {
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    return;
  }

  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(image, 0, 0, els.canvas.width, els.canvas.height);

  const names = getJointNames();
  for (const name of names) {
    const p = joints[name];
    if (!isFinitePoint(p)) continue;

    const [x, y] = p;
    const [cx, cy] = imageToCanvas(x, y);

    const isSelected = name === selectedJoint;
    ctx.beginPath();
    ctx.arc(cx, cy, pointRadius, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? "rgba(96,165,250,0.85)" : "rgba(230,232,242,0.75)";
    ctx.fill();

    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.strokeStyle = isSelected ? "rgba(96,165,250,1.0)" : "rgba(0,0,0,0.65)";
    ctx.stroke();

    ctx.font = `${Math.max(10, Math.round(12 * zoom))}px ui-monospace, monospace`;
    ctx.fillStyle = isSelected ? "rgba(96,165,250,1.0)" : "rgba(0,0,0,0.75)";
    ctx.fillText(name, cx + pointRadius + 4, cy - pointRadius - 2);
  }
};

// Events
els.btnLoadDefault.addEventListener("click", async () => {
  try {
    await loadImageFromUrl(DEFAULT_IMAGE);
  } catch (e) {
    setStatus(`加载默认图片失败：${String(e)}`, "err");
  }
});

els.btnLoadMain.addEventListener("click", async () => {
  try {
    await loadJointsFromUrl(DEFAULT_JOINTS);
  } catch (e) {
    setStatus(`加载 main_joint.json 失败：${String(e)}`, "err");
  }
});

els.btnLoadNames.addEventListener("click", async () => {
  try {
    await loadNamesFromUrl(DEFAULT_NAMES);
  } catch (e) {
    setStatus(`加载 button_name.json 失败：${String(e)}`, "err");
  }
});

els.fileImage.addEventListener("change", async (evt) => {
  const f = evt.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  try {
    await loadImageFromUrl(url);
  } catch (e) {
    setStatus(`加载图片失败：${String(e)}`, "err");
  } finally {
    URL.revokeObjectURL(url);
  }
});

els.fileJoints.addEventListener("change", async (evt) => {
  const f = evt.target.files?.[0];
  if (!f) return;
  try {
    const text = await f.text();
    const obj = JSON.parse(text);
    joints = normalizeJoints2D(obj);
    baselineJoints = JSON.parse(JSON.stringify(joints));
    undoStack = [];
    renderJointSelect();
    renderJointList();
    draw();
    setStatus(`关节已从文件加载：${Object.keys(joints).length} 个`, "ok");
  } catch (e) {
    setStatus(`读取 joints JSON 失败：${String(e)}`, "err");
  }
});

els.zoom.addEventListener("input", () => setZoom(Number(els.zoom.value)));
els.radius.addEventListener("input", () => setRadius(Number(els.radius.value)));

els.search.addEventListener("input", () => renderJointList());

els.jointSelect.addEventListener("change", () => setSelectedJoint(els.jointSelect.value));

els.btnSetCoord.addEventListener("click", () => {
  if (!selectedJoint) {
    setStatus("请先选择一个关节", "err");
    return;
  }
  const x = Number(els.coordX.value);
  const y = Number(els.coordY.value);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    setStatus("坐标无效", "err");
    return;
  }
  setJointPoint(selectedJoint, x, y);
  setStatus(`已设置 ${selectedJoint}: ${Math.round(x)}, ${Math.round(y)}`, "ok");
});

els.btnAdd.addEventListener("click", () => {
  const name = (els.newName.value || "").trim();
  if (!name) return;
  addJoint(name);
  els.newName.value = "";
});

els.btnUndo.addEventListener("click", () => undo());
els.btnReset.addEventListener("click", () => resetJoints());
els.btnDownload.addEventListener("click", () => downloadJSON());
els.btnCopy.addEventListener("click", () => copyJSON());

els.canvas.addEventListener("mousemove", (evt) => {
  const [xC, yC] = getMousePosCanvas(evt);
  const [xI, yI] = canvasToImage(xC, yC);
  els.cursor.textContent = `x: ${Math.round(xI)}, y: ${Math.round(yI)}`;

  if (dragging && dragJointName) {
    const [dx, dy] = dragOffset;
    const nx = xI + dx;
    const ny = yI + dy;
    joints[dragJointName] = [nx, ny];
    if (dragJointName === selectedJoint) {
      els.coordX.value = String(Math.round(nx));
      els.coordY.value = String(Math.round(ny));
    }
    draw();
  }
});

els.canvas.addEventListener("mousedown", (evt) => {
  if (evt.button !== 0) return;
  const [xC, yC] = getMousePosCanvas(evt);

  const near = findJointNearCanvas(xC, yC, Math.max(10, pointRadius + 6));
  if (near && isFinitePoint(joints[near])) {
    // drag existing
    dragging = true;
    dragJointName = near;
    dragPrevPoint = [...joints[near]];
    setSelectedJoint(near);

    const [xI, yI] = canvasToImage(xC, yC);
    const [jx, jy] = joints[near];
    dragOffset = [jx - xI, jy - yI];

    // record undo baseline for drag as a set op on mouseup
    evt.preventDefault();
    return;
  }

  // click-to-set selected joint
  if (!selectedJoint) {
    setStatus("没有可选关节（请先加载 joints JSON 或导入关节名）", "err");
    return;
  }

  const [xI, yI] = canvasToImage(xC, yC);
  setJointPoint(selectedJoint, xI, yI);
  setStatus(`已设置 ${selectedJoint}: ${Math.round(xI)}, ${Math.round(yI)}`, "ok");
});

window.addEventListener("mouseup", () => {
  if (!dragging || !dragJointName) return;

  const name = dragJointName;
  dragging = false;
  dragJointName = null;
  const prev = dragPrevPoint ? [...dragPrevPoint] : undefined;
  dragPrevPoint = null;

  // convert the last undo entry into a single set() change if it was a drag
  // We already mutated joints continuously; so we push a single undo here.
  const current = joints[name];
  // Avoid duplicating if prev equals current
  if (!(prev && current && prev[0] === current[0] && prev[1] === current[1])) {
    pushUndo({ type: "set", name, prev, next: [...current] });
  }
  setStatus(`已移动 ${name}`, "ok");
  renderJointList();
});

// zoom via wheel
els.canvas.addEventListener(
  "wheel",
  (evt) => {
    evt.preventDefault();
    const delta = Math.sign(evt.deltaY);
    const step = 0.08;
    setZoom(zoom * (delta > 0 ? 1 - step : 1 + step));
  },
  { passive: false }
);

// Init
(async function init() {
  setZoom(1.0);
  setRadius(7);

  try {
    await loadImageFromUrl(DEFAULT_IMAGE);
  } catch {
    setStatus("默认图片未加载（可手动选择图片文件）", "err");
  }

  try {
    await loadJointsFromUrl(DEFAULT_JOINTS);
  } catch {
    setStatus("默认 joints 未加载（可点击加载 main_joint.json 或导入文件）", "err");
  }
})();
