"use strict";

/* ---------------------------------------------------------------- classes */
const CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
  "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line", "congestion",
  "road_obstacle", "fire_smoke"];

// Colour encodes the family only (4 validated slots); the lane label carries the class.
const FAMILY = {
  collision: { name: "Collision", color: "var(--s2)" },
  vehicle: { name: "Vehicle violation", color: "var(--s1)" },
  pedestrian: { name: "Pedestrian", color: "var(--s3)" },
  flow: { name: "Flow and obstruction", color: "var(--s4)" },
};
const CLASS_FAMILY = {
  accident: "collision", near_miss: "collision", red_light: "vehicle", wrong_way: "vehicle",
  illegal_u_turn: "vehicle", illegal_turn: "vehicle", solid_line_crossing: "vehicle", stop_line: "vehicle",
  failure_to_yield: "vehicle", jaywalking: "pedestrian", stopped_vehicle: "flow", congestion: "flow",
  road_obstacle: "flow", fire_smoke: "flow",
};
const RULES = [
  ["accident", "Two road users' boxes touch on the ground and at least one drops from moving to below a third of its speed within about a second; both are then seen standing still for 5 s.", "First contact to both stopped"],
  ["red_light", "A vehicle's front crosses the stop line from the approach side while the signal has been red for at least 1 s, and it does not stop just past the line.", "Crossing to leaving the frame (at most 6 s)"],
  ["stop_line", "A vehicle that came from behind the line stands for 2 s or more between the stop line and the far edge of the crossing, on red.", "Stops to signal turns green"],
  ["wrong_way", "For 2 s or more and 3 sizes of travel, the vehicle drives against the dominant heading of the cells it is in (cos < −0.5); the cell must be coherent, and the vehicle's own votes are removed.", "First to last opposing sample"],
  ["illegal_u_turn", "Heading turns by 150° or more within 20 s, with no track teleports. Legality cannot be read from signs, so every U-turn is reported.", "Heading leaves the start direction to reaching the opposite one"],
  ["stopped_vehicle", "Stationary 10 s or more on the learned carriageway while at least two vehicles overtake it; queues get no overtakers. Anything stationary 90 s counts regardless.", "Stops to moves again or disappears"],
  ["jaywalking", "A pedestrian (not a rider) stands well inside the carriageway, outside the crossings grown by half a person's height, for 1.5 s or more.", "Steps onto the road to leaves it"],
  ["failure_to_yield", "A vehicle drives through a crossing without slowing while a pedestrian is on the carriageway part of the same crossing within 3 vehicle sizes.", "Enters to leaves the crossing"],
  ["congestion", "In one traffic direction, 6 or more vehicles with 80% below crawling speed for 45 s, lasting through at least 15 s of green (90 s if the signal is unreadable).", "Queue stops moving to clears"],
  ["road_obstacle", "A confidently detected animal (median confidence ≥ 0.5) on the carriageway for 2 s, not sitting on a person's box.", "Appears to leaves the road"],
];

/* ------------------------------------------------------------------ utils */
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of children) if (c != null) n.append(c);
  return n;
}
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const pretty = (label) => label.replace(/_/g, " ");
const colorOf = (label) => FAMILY[CLASS_FAMILY[label] || "flow"].color;

const tip = $("tooltip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${evt.clientY + 14}px`;
}
const hideTip = () => tip.classList.remove("on");

function niceTicks(max, n = 6) {
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function measure(container) { return Math.max(320, container.clientWidth || 640); }

/* --------------------------------------------------------------- timeline */
function timeline(container, events, duration, onSeek) {
  container.innerHTML = "";
  const labels = CLASSES.filter((c) => events.some((e) => e.label === c));
  const W = measure(container), left = Math.min(150, W * 0.3), right = 10, lane = 26, top = 6;
  const H = top + Math.max(1, labels.length) * lane + 24;
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Timeline of ${events.length} events` }, container);
  const x = (t) => left + (t / Math.max(duration, 1e-6)) * (W - left - right);
  const baseY = top + Math.max(1, labels.length) * lane;
  for (const v of niceTicks(duration, Math.max(3, Math.floor(W / 110)))) {
    svg("line", { class: "grid", x1: x(v), x2: x(v), y1: top, y2: baseY }, s);
    svg("text", { x: x(v), y: baseY + 16, "text-anchor": "middle" }, s).textContent = fmt(v);
  }
  if (!labels.length) {
    svg("text", { x: left, y: top + 17 }, s).textContent = "no events";
  }
  labels.forEach((lbl, i) => {
    const y = top + i * lane;
    svg("text", { class: "lane-label", x: left - 8, y: y + 17, "text-anchor": "end" }, s).textContent = pretty(lbl);
    for (const ev of events.filter((e) => e.label === lbl)) {
      const w = Math.max(3, x(ev.end) - x(ev.start));
      svg("rect", { x: x(ev.start), y: y + 5, width: w, height: lane - 10, rx: 4, fill: colorOf(lbl) }, s);
      const hit = svg("rect", { class: "hit", x: x(ev.start) - 4, y: y + 1, width: w + 8, height: lane - 2 }, s);
      hit.addEventListener("mousemove", (e) => showTip(e,
        `<b>${pretty(lbl)}</b><br>${fmt(ev.start)} to ${fmt(ev.end)} (${(ev.end - ev.start).toFixed(1)} s)`));
      hit.addEventListener("mouseleave", hideTip);
      hit.addEventListener("click", () => onSeek && onSeek(ev.start));
    }
  });
  svg("line", { class: "axis", x1: left, x2: W - right, y1: baseY, y2: baseY }, s);
  const cursor = svg("line", { class: "cursor", x1: left, x2: left, y1: top, y2: baseY, visibility: "hidden" }, s);
  const fams = [...new Set(labels.map((l) => CLASS_FAMILY[l]))];
  if (fams.length > 1) {
    const leg = el("div", { class: "legend" });
    for (const f of fams) leg.append(el("span", {}, el("i", { class: "sq", style: `background:${FAMILY[f].color}` }), FAMILY[f].name));
    container.append(leg);
  }
  return { setCursor(t) { cursor.setAttribute("visibility", "visible"); cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); } };
}

/* ------------------------------------------------------------- line chart */
function lineChart(container, { series, xMax, yMax, threshold, onSeek, yFormat = (v) => v, height = 180 }) {
  container.innerHTML = "";
  const W = measure(container), H = height, l = 40, r = 10, t = 8, b = 24;
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const all = series.flatMap((se) => se.points);
  xMax = xMax ?? Math.max(1, ...all.map((p) => p[0]));
  yMax = yMax ?? Math.max(1e-6, ...all.map((p) => p[1])) * 1.1;
  const x = (v) => l + (v / xMax) * (W - l - r), y = (v) => t + (1 - v / yMax) * (H - t - b);
  for (const v of niceTicks(yMax, 4)) {
    if (v > yMax) continue;
    svg("line", { class: "grid", x1: l, x2: W - r, y1: y(v), y2: y(v) }, s);
    svg("text", { x: l - 6, y: y(v) + 4, "text-anchor": "end" }, s).textContent = yFormat(v);
  }
  for (const v of niceTicks(xMax, Math.max(3, Math.floor(W / 110))))
    svg("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, s).textContent = fmt(v);
  svg("line", { class: "axis", x1: l, x2: W - r, y1: y(0), y2: y(0) }, s);
  if (threshold != null) {
    svg("line", { class: "threshold", x1: l, x2: W - r, y1: y(threshold), y2: y(threshold) }, s);
    svg("text", { x: W - r, y: y(threshold) - 4, "text-anchor": "end" }, s).textContent = `alarm ${threshold}`;
  }
  for (const se of series) {
    if (!se.points.length) continue;
    const d = se.points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    svg("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round" }, s);
  }
  const cursor = svg("line", { class: "cursor", y1: t, y2: H - b, visibility: "hidden" }, s);
  const hover = svg("line", { class: "cursor", y1: t, y2: H - b, visibility: "hidden", "stroke-dasharray": "2 2" }, s);
  const hit = svg("rect", { class: "hit", x: l, y: t, width: W - l - r, height: H - t - b }, s);
  const toT = (e) => {
    const box = s.getBoundingClientRect();
    return Math.max(0, Math.min(xMax, ((e.clientX - box.left) * (W / box.width) - l) / (W - l - r) * xMax));
  };
  hit.addEventListener("mousemove", (e) => {
    const tt = toT(e);
    hover.setAttribute("visibility", "visible");
    hover.setAttribute("x1", x(tt)); hover.setAttribute("x2", x(tt));
    const rows = series.map((se) => {
      let best = null;
      for (const p of se.points) if (!best || Math.abs(p[0] - tt) < Math.abs(best[0] - tt)) best = p;
      return best ? `<i style="display:inline-block;width:10px;height:3px;background:${se.color};margin-right:6px;vertical-align:middle"></i>${se.name}: <b>${yFormat(best[1])}</b>` : "";
    });
    showTip(e, `${fmt(tt)}<br>${rows.join("<br>")}`);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); hover.setAttribute("visibility", "hidden"); });
  hit.addEventListener("click", (e) => onSeek && onSeek(toT(e)));
  if (series.length > 1) {
    const leg = el("div", { class: "legend" });
    for (const se of series) leg.append(el("span", {}, el("i", { style: `background:${se.color}` }), se.name));
    container.append(leg);
  }
  return { setCursor(tt) { cursor.setAttribute("visibility", "visible"); cursor.setAttribute("x1", x(tt)); cursor.setAttribute("x2", x(tt)); } };
}

function riskChart(container, risk, duration, onSeek) {
  if (!risk.length) { container.textContent = "No risk curve."; return { setCursor() {} }; }
  return lineChart(container, { series: [{ name: "risk", color: "var(--s1)", points: risk }], xMax: duration,
    yMax: 1, threshold: 0.5, onSeek, yFormat: (v) => (+v).toFixed(2) });
}

function bandChart(container, phases, duration) {
  container.innerHTML = "";
  const W = measure(container), H = 46, l = 40, r = 10;
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const x = (v) => l + (v / Math.max(duration, 1e-6)) * (W - l - r);
  const col = { RED: "var(--critical)", GREEN: "var(--good)", UNKNOWN: "var(--grid)" };
  for (const [a, b, st] of phases) {
    const w = Math.max(1, x(b) - x(a));
    svg("rect", { x: x(a), y: 4, width: w, height: 20, fill: col[st] || col.UNKNOWN }, s);
    if (w > 46) svg("text", { x: x(a) + 6, y: 18, style: "fill:#fff" }, s).textContent = st.toLowerCase();
    const hit = svg("rect", { class: "hit", x: x(a), y: 2, width: w, height: 24 }, s);
    hit.addEventListener("mousemove", (e) => showTip(e, `<b>${st}</b> ${fmt(a)} to ${fmt(b)} (${(b - a).toFixed(1)} s)`));
    hit.addEventListener("mouseleave", hideTip);
  }
  for (const v of niceTicks(duration, Math.max(3, Math.floor(W / 110))))
    svg("text", { x: x(v), y: H - 4, "text-anchor": "middle" }, s).textContent = fmt(v);
}

function barChart(container, { labels, values, color = "var(--s1)", format = (v) => v, horizontal = false }) {
  container.innerHTML = "";
  const W = measure(container);
  if (horizontal) {
    const rowH = 26, l = Math.min(160, W * 0.35), r = 40, H = labels.length * rowH + 6;
    const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    const max = Math.max(1, ...values);
    labels.forEach((lab, i) => {
      const y = 3 + i * rowH, w = (values[i] / max) * (W - l - r);
      svg("text", { class: "lane-label", x: l - 8, y: y + 17, "text-anchor": "end" }, s).textContent = lab;
      svg("rect", { x: l, y: y + 4, width: Math.max(2, w), height: rowH - 8, rx: 4,
        fill: typeof color === "function" ? color(i) : color }, s);
      svg("text", { x: l + w + 6, y: y + 17 }, s).textContent = format(values[i]);
    });
    return;
  }
  const H = 170, l = 36, r = 8, t = 8, b = 22;
  const s = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const max = Math.max(1, ...values), bw = (W - l - r) / values.length;
  const y = (v) => t + (1 - v / max) * (H - t - b);
  for (const v of niceTicks(max, 4)) {
    if (v > max) continue;
    svg("line", { class: "grid", x1: l, x2: W - r, y1: y(v), y2: y(v) }, s);
    svg("text", { x: l - 6, y: y(v) + 4, "text-anchor": "end" }, s).textContent = v;
  }
  values.forEach((v, i) => {
    svg("rect", { x: l + i * bw + 1, y: y(v), width: Math.max(1, bw - 2), height: y(0) - y(v), rx: 2, fill: color }, s);
    const hit = svg("rect", { class: "hit", x: l + i * bw, y: t, width: bw, height: H - t - b }, s);
    hit.addEventListener("mousemove", (e) => showTip(e, `${labels[i]}: <b>${format(v)}</b>`));
    hit.addEventListener("mouseleave", hideTip);
  });
  for (let i = 0; i < labels.length; i += Math.ceil(labels.length / 8))
    svg("text", { x: l + i * bw, y: H - 6 }, s).textContent = labels[i].split("–")[0];
  svg("line", { class: "axis", x1: l, x2: W - r, y1: y(0), y2: y(0) }, s);
}

/* ------------------------------------------------------------ event cards */
function eventCards(container, events, onSeek, thumbBase = "") {
  container.innerHTML = "";
  if (!events.length) { container.append(el("p", { class: "muted" }, "No events.")); return; }
  for (const ev of events) {
    const src = ev.thumb ? (ev.thumb.startsWith("data:") ? ev.thumb : thumbBase + ev.thumb) : null;
    container.append(el("button", { class: "event", type: "button", onclick: () => onSeek(ev.start) },
      src ? el("img", { src, alt: `${pretty(ev.label)} at ${fmt(ev.start)}`, loading: "lazy" }) : null,
      el("div", { class: "body" },
        el("div", { class: "lbl" }, el("i", { style: `background:${colorOf(ev.label)}` }), pretty(ev.label)),
        el("div", { class: "when" }, `${fmt(ev.start)} to ${fmt(ev.end)}, ${(ev.end - ev.start).toFixed(1)} s`))));
  }
}

/* A video plus the charts that follow its clock. */
function linkPlayer(video, charts) {
  const tick = () => charts.forEach((c) => c && c.setCursor(video.currentTime));
  video.addEventListener("timeupdate", tick);
  return (t) => { video.currentTime = t; video.play().catch(() => {}); };
}

/* --------------------------------------------------------------- live demo */
const demo = { result: null, url: null };

function setupDemo() {
  const drop = $("drop"), input = $("file");
  ["dragenter", "dragover"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && startDemo(e.dataTransfer.files[0]));
  input.addEventListener("change", () => input.files[0] && startDemo(input.files[0]));
  $("demoDownload").addEventListener("click", () => {
    if (!demo.result) return;
    const blob = new Blob([JSON.stringify({ events: demo.result.official }, null, 1)], { type: "application/json" });
    el("a", { href: URL.createObjectURL(blob), download: "events.json" }).click();
  });
}

function progress(label, frac) {
  $("progress").hidden = false;
  $("progLabel").textContent = label;
  $("progPct").textContent = `${Math.round(frac * 100)}%`;
  $("progBar").style.width = `${frac * 100}%`;
}
function demoError(msg) { const e = $("demoError"); e.hidden = false; e.textContent = msg; $("progress").hidden = true; }

function startDemo(file) {
  $("demoError").hidden = true;
  $("demoResult").hidden = true;
  if (!/\.mp4$/i.test(file.name)) return demoError("Please choose an .mp4 file.");
  if (file.size > 500 * 2 ** 20) return demoError("The demo accepts files up to 500 MB.");
  $("dropText").textContent = file.name;
  if (demo.url) URL.revokeObjectURL(demo.url);
  demo.url = URL.createObjectURL(file);
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "api/jobs");
  xhr.upload.onprogress = (e) => e.lengthComputable && progress("Uploading", e.loaded / e.total);
  xhr.onerror = () => demoError("Upload failed: the server could not be reached.");
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status !== 200) return demoError(body.detail || `Upload failed (HTTP ${xhr.status}).`);
    poll(body.id);
  };
  progress("Uploading", 0);
  xhr.send(form);
}

async function poll(id) {
  try {
    const r = await fetch(`api/jobs/${id}`);
    const j = await r.json();
    if (!r.ok) return demoError(j.detail || "Job lost.");
    if (j.state === "error") return demoError(`Analysis failed: ${j.error}`);
    if (j.state === "done") { progress("Done", 1); return showDemo(j.result); }
    progress(j.state === "queued" ? "Queued: another video is being analysed" : `Analysing: ${j.stage}`, j.progress || 0);
    setTimeout(() => poll(id), 1000);
  } catch (e) {
    setTimeout(() => poll(id), 2000);
  }
}

function showDemo(res) {
  demo.result = res;
  $("demoResult").hidden = false;
  const video = $("demoVideo"), canvas = $("demoCanvas");
  video.src = demo.url;
  const dur = res.video.duration;
  const tl = timeline($("demoTimeline"), res.events, dur, (t) => seek(t));
  const rc = riskChart($("demoRisk"), res.risk, dur, (t) => seek(t));
  const seek = linkPlayer(video, [tl, rc]);
  eventCards($("demoEvents"), res.events, seek);
  $("demoNote").textContent = `${res.events.length} events in ${fmt(dur)} of video; analysed in ${res.seconds} s ` +
    `with ${res.profile.weights} at ${res.profile.analysis_fps} fps. Boxes are drawn from our tracks; highlighted ones are event actors.`;
  canvas.hidden = false;
  video.onerror = () => {
    canvas.hidden = true;
    $("demoNote").textContent += " Your browser cannot play this file's codec, so there is no playback; the timeline, risk curve and events are complete.";
  };
  drawOverlayLoop(video, canvas, res);
}

function drawOverlayLoop(video, canvas, res) {
  demo.frameLoop = (demo.frameLoop || 0) + 1;
  const loopId = demo.frameLoop;
  const ctx = canvas.getContext("2d"), ov = res.overlay;
  const catColor = ["#3cbeeb", "#6edc5a", "#ffc800"];
  const draw = () => {
    if (loopId !== demo.frameLoop) return;          // a newer upload replaced this one
    const w = video.clientWidth, h = video.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    const t = video.currentTime;
    // letterboxing inside the <video> element
    const ar = res.video.width / res.video.height, vw = Math.min(w, h * ar), vh = vw / ar;
    const ox = (w - vw) / 2, oy = (h - vh) / 2;
    let lo = 0, hi = ov.t.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (ov.t[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(ov.t[lo - 1] - t) < Math.abs(ov.t[lo] - t)) lo -= 1;
    const active = res.events.filter((e) => e.start <= t && t <= e.end);
    const hl = new Map();
    active.forEach((e) => e.actors.forEach((a) => hl.set(a, e.label)));
    if (ov.t.length && Math.abs(ov.t[lo] - t) < 0.3) {
      ctx.font = "12px system-ui, sans-serif";
      for (const [x1, y1, x2, y2, id, cat] of ov.boxes[lo]) {
        const X = ox + (x1 / 1e4) * vw, Y = oy + (y1 / 1e4) * vh, BW = ((x2 - x1) / 1e4) * vw, BH = ((y2 - y1) / 1e4) * vh;
        const label = hl.get(id);
        ctx.strokeStyle = label ? "#ff3b3b" : catColor[cat];
        ctx.lineWidth = label ? 3 : 1;
        ctx.strokeRect(X, Y, BW, BH);
        if (label) {
          const text = `${id} ${pretty(label)}`;
          ctx.fillStyle = "#ff3b3b";
          ctx.fillRect(X, Y - 16, ctx.measureText(text).width + 8, 16);
          ctx.fillStyle = "#fff";
          ctx.fillText(text, X + 4, Y - 4);
        }
      }
    }
    active.forEach((e, k) => {
      ctx.font = "bold 14px system-ui, sans-serif";
      const text = pretty(e.label).toUpperCase(), tw = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(208,59,59,0.92)";
      ctx.fillRect(ox + 10, oy + 10 + k * 30, tw + 16, 24);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, ox + 18, oy + 27 + k * 30);
    });
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

/* ----------------------------------------------------- samples & dashboard */
async function loadJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url); return r.json(); }

async function setupSamples() {
  let manifest;
  try { manifest = await loadJSON("data/manifest.json"); } catch (_) { manifest = { videos: [] }; }
  const vids = manifest.videos || [];
  if (!vids.length) { $("noSamples").hidden = false; $("failures").textContent = "Built with the sample results."; }
  const datas = await Promise.all(vids.map((v) => loadJSON(`data/${v.dir}/data.json`).catch(() => null)));
  const ok = vids.map((v, i) => [v, datas[i]]).filter(([, d]) => d);
  tabs($("sampleTabs"), ok, ([v, d]) => showSample(v, d));
  tabs($("edaTabs"), ok, ([v, d]) => showEda(v, d));
  dashboard(ok);
  heroTiles(ok);
  failures();
}

function tabs(container, items, onPick) {
  container.innerHTML = "";
  items.forEach((it, i) => {
    const b = el("button", { role: "tab", "aria-selected": i === 0 ? "true" : "false", type: "button" }, it[0].video);
    b.addEventListener("click", () => {
      container.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", "false"));
      b.setAttribute("aria-selected", "true");
      onPick(it);
    });
    container.append(b);
  });
  if (items.length) onPick(items[0]);
}

function showSample(v, d) {
  $("sampleView").hidden = false;
  const video = $("sampleVideo");
  video.src = d.annotated ? `data/${v.dir}/${d.annotated}` : "";
  const dur = d.meta.duration;
  const tl = timeline($("sampleTimeline"), d.events, dur, (t) => seek(t));
  const rc = riskChart($("sampleRisk"), d.risk, dur, (t) => seek(t));
  const seek = linkPlayer(video, [tl, rc]);
  eventCards($("sampleEvents"), d.events, seek, `data/${v.dir}/`);
}

function dashboard(items) {
  const counts = Object.fromEntries(CLASSES.map((c) => [c, 0]));
  let dur = 0, n = 0;
  for (const [, d] of items) { dur += d.meta.duration; for (const e of d.events) { counts[e.label] += 1; n += 1; } }
  const t = $("dashTiles");
  t.innerHTML = "";
  const tile = (v, k) => t.append(el("div", { class: "tile" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)));
  tile(items.length, "sample videos");
  tile(fmt(dur), "of video");
  tile(n, "events detected");
  tile(dur ? (n / dur * 3600).toFixed(0) : "–", "events per hour");
  const present = CLASSES.filter((c) => counts[c]);
  if (present.length) barChart($("dashClasses"), { labels: present.map(pretty), values: present.map((c) => counts[c]),
    color: (i) => colorOf(present[i]), horizontal: true });
  else $("dashClasses").textContent = "No events yet.";
  const gal = $("classGallery");
  const firsts = [];
  for (const c of present) for (const [v, d] of items) {
    const e = d.events.find((x) => x.label === c);
    if (e) { firsts.push({ ...e, thumb: `data/${v.dir}/${e.thumb}`, video: v }); break; }
  }
  eventCards(gal, firsts, () => { location.hash = "#results"; });
}

function heroTiles(items) {
  const t = $("heroTiles");
  const tile = (v, k) => t.append(el("div", { class: "tile" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)));
  tile("10 of 14", "classes emitted");
  tile("37", "unit and end-to-end tests");
  tile("0.95×", "real time, 4K on a 4-core CPU");
  tile(items.length || "–", "sample videos annotated");
}

async function failures() {
  let list = [];
  try { list = await loadJSON("failures.json"); } catch (_) { /* optional */ }
  const box = $("failures");
  box.innerHTML = "";
  if (!list.length) { box.append(el("p", { class: "muted" }, "Failure cases are listed once the dev labels are scored (scripts/dev_eval.py).")); return; }
  const ul = el("ul");
  for (const f of list) ul.append(el("li", {}, el("strong", {}, `${f.video} ${f.time || ""}: `), f.text));
  box.append(ul);
}

/* --------------------------------------------------------------------- EDA */
function showEda(v, d) {
  $("edaView").hidden = false;
  const m = d.meta;
  const rows = [["Resolution", `${m.width} × ${m.height}`], ["Frame rate", `${m.fps} fps`], ["Duration", `${fmt(m.duration)} (${m.frames} frames)`],
    ["File size", `${m.size_mb} MB`], ["Tracks", Object.entries(m.tracks).map(([k, n]) => `${n} ${k}`).join(", ")],
    ["Detected classes", Object.entries(m.classes).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${n}`).join(", ")],
    ["Signal readable", d.signal_observable ? "yes (red and green seen)" : "no"],
    ["Main traffic directions", d.directions.map((x) => `${x.angle_deg}°`).join(", ") || "–"]];
  const tb = $("edaMeta");
  tb.innerHTML = "";
  for (const [k, val] of rows) tb.append(el("tr", {}, el("th", {}, k), el("td", {}, val)));
  const sec = (arr) => arr.map((y, i) => [i, y]);
  lineChart($("edaCounts"), { series: [
    { name: "cars", color: "var(--s1)", points: sec(d.counts.car) },
    { name: "buses and trucks", color: "var(--s2)", points: sec(d.counts.bus_truck) },
    { name: "two-wheelers", color: "var(--s3)", points: sec(d.counts.two_wheeler) },
    { name: "people", color: "var(--s4)", points: sec(d.counts.person) }], xMax: m.duration, yFormat: (x) => (+x).toFixed(0) });
  bandChart($("edaSignal"), d.signal, m.duration);
  lineChart($("edaBright"), { series: [{ name: "brightness", color: "var(--s1)", points: sec(d.brightness).filter((p) => p[1] > 0) }],
    xMax: m.duration, yMax: 255, yFormat: (x) => (+x).toFixed(0), height: 150 });
  const e = d.speed_hist.edges;
  if (!d.speed_hist.counts.some((c) => c > 0)) $("edaSpeed").textContent = "No moving vehicles in this video.";
  else barChart($("edaSpeed"), { labels: d.speed_hist.counts.map((_, i) => `${e[i]}–${e[i + 1]}`), values: d.speed_hist.counts });
  const g = $("edaImages");
  g.innerHTML = "";
  const imgs = [["background.jpg", "Median of 40 frames: the empty road, with moving traffic removed."],
    ["heatmap.jpg", "Where vehicles touch the ground: lanes and the turning paths through the intersection."],
    ["heatmap_people.jpg", "Where people walk and wait: crossings, kerbs and the central island."],
    ["trajectories.jpg", "Every track; hue is the direction of travel."],
    ["flow.jpg", "Learned carriageway (tint) and dominant heading per cell (arrows): the scene model the rules use."]];
  for (const [f, cap] of imgs) g.append(el("figure", {}, el("img", { src: `data/${v.dir}/${f}`, alt: cap, loading: "lazy" }), el("figcaption", {}, cap)));
}

/* ----------------------------------------------------------- approach, team */
function diagram() {
  const steps = [["Video", "4K, 25–30 fps", false], ["Frame reader", "strided, prefetched", false],
    ["YOLOv8", "COCO-pretrained", true], ["ByteTrack", "per category", false], ["Smooth + stitch", "offline tracks", false],
    ["Scene model", "lanes, carriageway", true], ["Rules", "10 classes", false], ["Merge", "segments", false]];
  const W = 1060, H = 190, bw = 118, gap = (W - steps.length * bw) / (steps.length - 1);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Pipeline diagram" });
  const defs = svg("defs", {}, s);
  const mk = svg("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
  svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--muted)" }, mk);
  steps.forEach(([name, sub, learned], i) => {
    const x = i * (bw + gap);
    svg("rect", { class: `box${learned ? " learned" : ""}`, x, y: 20, width: bw, height: 54, rx: 8 }, s);
    svg("text", { x: x + bw / 2, y: 43, "text-anchor": "middle" }, s).textContent = name;
    svg("text", { class: "sub", x: x + bw / 2, y: 61, "text-anchor": "middle" }, s).textContent = sub;
    if (i) svg("path", { class: "arrow", d: `M${x - gap + 2},47 L${x - 3},47` }, s);
  });
  svg("text", { x: W - bw / 2, y: 100, "text-anchor": "middle" }, s).textContent = "Part A events";
  svg("path", { class: "arrow", d: `M${W - bw / 2},74 L${W - bw / 2},86` }, s);
  // Part B lane
  const y2 = 124, partB = [["Every frame", "harness order", false], ["YOLOv8 + ByteTrack", "own, causal", true],
    ["Closest approach", "every pair", false], ["Hard braking", "near others", false], ["Hold + map", "P(accident ≤ 5 s)", false]];
  const bw2 = 150, gap2 = (W - partB.length * bw2) / (partB.length - 1);
  partB.forEach(([name, sub, learned], i) => {
    const x = i * (bw2 + gap2);
    svg("rect", { class: `box${learned ? " learned" : ""}`, x, y: y2, width: bw2, height: 54, rx: 8 }, s);
    svg("text", { x: x + bw2 / 2, y: y2 + 23, "text-anchor": "middle" }, s).textContent = name;
    svg("text", { class: "sub", x: x + bw2 / 2, y: y2 + 41, "text-anchor": "middle" }, s).textContent = sub;
    if (i) svg("path", { class: "arrow", d: `M${x - gap2 + 2},${y2 + 27} L${x - 3},${y2 + 27}` }, s);
  });
  svg("text", { class: "sub", x: 0, y: 12 }, s).textContent = "Part A: offline, whole video (blue outline = learned model)";
  svg("text", { class: "sub", x: 0, y: y2 - 8 }, s).textContent = "Part B: causal, frame by frame, never sees Part A";
  $("diagram").append(s);
  const tb = $("rulesTable").querySelector("tbody");
  for (const [c, when, seg] of RULES) tb.append(el("tr", {}, el("td", {}, el("code", {}, c)), el("td", {}, when), el("td", {}, seg)));
}

async function team() {
  let data = { members: [], links: [] };
  try { data = await loadJSON("team.json"); } catch (_) { /* optional */ }
  const box = $("teamCards");
  for (const m of data.members) {
    const links = el("div", { class: "links" });
    for (const [k, url] of Object.entries(m.links || {})) if (url) links.append(el("a", { href: url, rel: "noopener" }, k));
    box.append(el("div", { class: "member" }, el("h3", {}, m.name), el("div", { class: "role" }, m.role),
      el("ul", {}, ...(m.did || []).map((x) => el("li", {}, x))),
      m.projects && m.projects.length ? el("p", { class: "muted" }, `Proud of: ${m.projects.join("; ")}`) : null, links,
      m.todo ? el("p", { class: "todo" }, m.todo) : null));
  }
  const ul = $("links");
  for (const [k, url] of data.links) ul.append(el("li", {}, el("a", { href: url }, k)));
}

/* -------------------------------------------------------------- label tool */
function setupLabel() {
  const input = $("labelFile"), video = $("labelVideo"), sel = $("labelClass");
  const state = { events: [], pending: null, name: "" };
  CLASSES.forEach((c) => sel.append(el("option", { value: c }, c)));
  const redraw = () => {
    const dur = video.duration || 1;
    timeline($("labelTimeline"), state.events, dur, (t) => { video.currentTime = t; });
    const list = $("labelList");
    list.innerHTML = "";
    state.events.forEach((ev, i) => list.append(el("div", { class: "event" }, el("div", { class: "body" },
      el("button", { class: "rm", type: "button", "aria-label": "remove", onclick: () => { state.events.splice(i, 1); redraw(); } }, "✕"),
      el("div", { class: "lbl" }, el("i", { style: `background:${colorOf(ev.label)}` }), pretty(ev.label)),
      el("div", { class: "when" }, `${fmt(ev.start)} to ${fmt(ev.end)}`)))));
    $("labelPending").textContent = state.pending ? `${state.pending.label} started at ${fmt(state.pending.start)}` : "";
  };
  const start = () => { state.pending = { label: sel.value, start: +video.currentTime.toFixed(2) }; redraw(); };
  const end = () => {
    if (!state.pending || video.currentTime <= state.pending.start) return;
    state.events.push({ ...state.pending, end: +video.currentTime.toFixed(2) });
    state.events.sort((a, b) => a.start - b.start);
    state.pending = null;
    redraw();
  };
  input.addEventListener("change", () => {
    const f = input.files[0];
    if (!f) return;
    state.name = f.name;
    video.src = URL.createObjectURL(f);
    $("labelUi").hidden = false;
    video.onloadedmetadata = redraw;
  });
  $("labelStart").addEventListener("click", start);
  $("labelEnd").addEventListener("click", end);
  document.addEventListener("keydown", (e) => {
    if ($("labelUi").hidden || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    const r = $("label").getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) return;
    if (e.key === "s" || e.key === "S") start();
    else if (e.key === "e" || e.key === "E") end();
    else if (e.key === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    else if (e.key === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - 1);
    else if (e.key === "ArrowRight") video.currentTime += 1;
  });
  $("labelExport").addEventListener("click", () => {
    const gt = { [state.name]: { duration: +video.duration.toFixed(3), fps: 25.0,
      events: state.events.map((e) => [e.start, e.end, e.label]) } };
    const blob = new Blob([JSON.stringify(gt, null, 1)], { type: "application/json" });
    el("a", { href: URL.createObjectURL(blob), download: "ground_truth.json" }).click();
  });
}

/* -------------------------------------------------------------------- boot */
setupDemo();
diagram();
team();
setupSamples();
setupLabel();
