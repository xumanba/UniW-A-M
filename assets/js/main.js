(function () {
  "use strict";
  const D = window.UNIWAM;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const KIND_VAR = { joint: "--c-joint", idm: "--c-idm", ac: "--c-ac", ao: "--c-ao", acwm: "--c-acwm", base: "--c-base", ours: "--accent" };
  const IFACE_KIND = { "Joint-WAM": "joint", "IDM-WAM": "idm", "Action-Centric-WAM": "ac", "Action-Only-WAM": "ao", "AC-WM": "acwm" };
  const col = (k) => css(KIND_VAR[k] || "--c-base");
  const short = (n) => n.replace("Action-Centric-WAM", "Action-Centric").replace("Action-Only-WAM", "Action-Only");
  const charts = {};

  /* ---------- segmented controls ---------- */
  function seg(el, items, onPick, initial = 0) {
    el.innerHTML = "";
    items.forEach((it, i) => {
      const b = document.createElement("button");
      b.textContent = it; b.type = "button";
      if (i === initial) b.classList.add("on");
      b.onclick = () => { $$("button", el).forEach((x) => x.classList.remove("on")); b.classList.add("on"); onPick(it, i); };
      el.appendChild(b);
    });
  }
  function bindStaticSeg(el, onPick) {
    $$("button", el).forEach((b) => b.addEventListener("click", () => {
      $$("button", el).forEach((x) => x.classList.remove("on")); b.classList.add("on"); onPick(b);
    }));
  }

  /* ---------- Chart.js setup ---------- */
  const valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.enabled) return;
      const { ctx } = chart; ctx.save();
      ctx.font = `600 ${opts.size || 11}px ${css("--font-body")}`;
      ctx.fillStyle = css("--text-2");
      chart.data.datasets.forEach((ds, di) => {
        const meta = chart.getDatasetMeta(di);
        if (meta.hidden) return;
        meta.data.forEach((el, i) => {
          const v = ds.data[i]; if (v == null) return;
          const txt = (opts.fmt || ((x) => x))(v);
          if (chart.options.indexAxis === "y") { ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(txt, el.x + 6, el.y); }
          else { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(txt, el.x, el.y - 4); }
        });
      });
      ctx.restore();
    }
  };
  const pointLabels = {
    id: "pointLabels",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.enabled) return;
      const { ctx } = chart; ctx.save();
      chart.data.datasets.forEach((ds, di) => {
        chart.getDatasetMeta(di).data.forEach((el, i) => {
          const p = ds.data[i]; if (!p.label) return;
          ctx.font = `${p.bold ? 700 : 500} 12px ${css("--font-body")}`;
          ctx.fillStyle = p.bold ? css("--accent") : css("--text-2");
          ctx.textAlign = p.align || "left"; ctx.textBaseline = "middle";
          ctx.fillText(p.label, el.x + (p.dx || 12), el.y + (p.dy || 0));
        });
      });
      ctx.restore();
    }
  };
  // Charts are optional: if Chart.js fails to load, videos and the rest of the page still work.
  const HAS_CHART = typeof Chart !== "undefined";
  if (HAS_CHART) Chart.register(valueLabels, pointLabels);

  function applyDefaults() {
    Chart.defaults.font.family = css("--font-body");
    Chart.defaults.font.size = 12;
    Chart.defaults.color = css("--text-2");
    Chart.defaults.borderColor = css("--grid");
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyle = "rectRounded";
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.boxHeight = 10;
    Chart.defaults.plugins.tooltip.backgroundColor = css("--tooltip-bg");
    Chart.defaults.plugins.tooltip.titleColor = css("--tooltip-fg");
    Chart.defaults.plugins.tooltip.bodyColor = css("--tooltip-fg");
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration = 500;
  }
  const radius = () => parseFloat(css("--bar-radius")) || 4;
  function make(id, cfg) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), cfg);
    return charts[id];
  }
  const pctAxis = (max = 100, title) => ({ min: 0, max, grid: { color: css("--grid") }, border: { display: false }, ticks: { callback: (v) => v + "%" }, title: title ? { display: true, text: title } : undefined });
  const noGrid = { grid: { display: false }, border: { display: false } };

  /* ---------- Real world ---------- */
  let rwIdx = 3; // 0..2 task, 3 = average
  function drawReal() {
    const rows = D.realWorld;
    const vals = rows.map((r) => (rwIdx === 3 ? r.avg : r.v[rwIdx]));
    make("chart-real", {
      type: "bar",
      data: { labels: rows.map((r) => r.name), datasets: [{ data: vals, backgroundColor: rows.map((r) => col(r.kind)), borderRadius: radius(), barPercentage: 0.72, categoryPercentage: 0.9 }] },
      options: {
        indexAxis: "y", layout: { padding: { right: 36 } },
        plugins: { legend: { display: false }, valueLabels: { enabled: true, fmt: (v) => (rwIdx === 3 ? v.toFixed(1) : v) + "%" } },
        scales: { x: pctAxis(100), y: { ...noGrid, ticks: { font: { weight: 500 } } } }
      }
    });
  }

  /* ---------- RoboTwin + LIBERO ---------- */
  let rtKey = "c2r";
  function drawRobotwin() {
    const rows = D.robotwin.slice().sort((a, b) => a[rtKey] - b[rtKey]);
    make("chart-robotwin", {
      type: "bar",
      data: { labels: rows.map((r) => r.name), datasets: [{ data: rows.map((r) => r[rtKey]), backgroundColor: rows.map((r) => (r.ours ? css("--accent") : css("--c-base"))), borderRadius: radius(), barPercentage: 0.7 }] },
      options: { indexAxis: "y", layout: { padding: { right: 44 } }, plugins: { legend: { display: false }, valueLabels: { enabled: true, fmt: (v) => v.toFixed(1) + "%" } }, scales: { x: pctAxis(100), y: noGrid } }
    });
  }
  function buildLibero() {
    const t = $("#tbl-libero"); const L = D.libero;
    const all = L.groups.flatMap((g) => g.rows);
    const rank = L.cols.map((_, c) => { const s = [...new Set(all.map((r) => r[c + 1]))].sort((a, b) => b - a); return [s[0], s[1]]; });
    let h = "<thead><tr><th>Model</th>" + L.cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
    L.groups.forEach((g) => {
      if (g.label) h += `<tr class="grp"><td colspan="${L.cols.length + 1}">${g.label}</td></tr>`;
      g.rows.forEach((r) => {
        h += `<tr class="${g.ours ? "ours" : ""}"><td>${r[0]}</td>` + r.slice(1).map((v, c) => {
          const cls = v === rank[c][0] ? "best" : v === rank[c][1] ? "second" : "";
          return `<td class="${cls}">${v.toFixed(1)}</td>`;
        }).join("") + "</tr>";
      });
    });
    t.innerHTML = h + "</tbody>";
  }

  /* ---------- WorldArena ---------- */
  let waTask = D.tasks[0];
  const WA_MODELS = ["Cosmos-Predict 2.5", "CtrlWorld", "UniW(A)M"];
  function drawWA() {
    const T = D.worldArena.tasks[waTask];
    const colors = [css("--c-base"), css("--muted"), css("--accent")];
    make("chart-wa", {
      type: "radar",
      data: {
        labels: D.worldArena.metrics.map((m) => m.split(" ")),
        datasets: WA_MODELS.map((m, i) => ({
          label: m === "CtrlWorld" ? "Ctrl-World" : m, data: T[m].slice(0, 5),
          borderColor: colors[i], backgroundColor: colors[i] + (i === 2 ? "33" : "14"),
          borderWidth: i === 2 ? 2.5 : 1.5, pointRadius: i === 2 ? 3.5 : 2.5, pointBackgroundColor: colors[i]
        }))
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: { r: { min: 0, max: 100, ticks: { stepSize: 25, backdropColor: "transparent", display: false }, grid: { color: css("--grid") }, angleLines: { color: css("--grid") }, pointLabels: { font: { size: 11, weight: 500 }, color: css("--text-2") } } }
      }
    });
  }
  function buildEWM() {
    const el = $("#ewm-list"); let h = "";
    D.tasks.forEach((t) => {
      const T = D.worldArena.tasks[t];
      h += `<div class="ewm-task"><div class="ewm-name">${t}</div>`;
      WA_MODELS.forEach((m) => {
        const v = T[m][5]; const ours = m === "UniW(A)M";
        h += `<div class="ewm-row ${ours ? "ours" : ""}"><span>${m === "CtrlWorld" ? "Ctrl-World" : m}</span><div class="ewm-bar"><i style="width:${(v / 70) * 100}%"></i></div><b>${v.toFixed(2)}</b></div>`;
      });
      h += "</div>";
    });
    el.innerHTML = h;
  }

  /* ---------- PAIR ---------- */
  let oodIdx = 3;
  function drawOOD() {
    const rows = D.ood;
    make("chart-ood", {
      type: "bar",
      data: { labels: rows.map((r) => short(r.name)), datasets: [{ data: rows.map((r) => (oodIdx === 3 ? r.avg : r.v[oodIdx])), backgroundColor: rows.map((r) => col(r.kind)), borderRadius: radius(), barPercentage: 0.66 }] },
      options: { plugins: { legend: { display: false }, valueLabels: { enabled: true, fmt: (v) => (oodIdx === 3 ? v.toFixed(1) : v) + "%" } }, scales: { y: pctAxis(100), x: noGrid } }
    });
  }
  let selTask = D.tasks[0];
  function drawSel() {
    const S = D.selection[selTask];
    make("chart-sel", {
      type: "bar",
      data: { labels: D.phases, datasets: D.interfaces.map((n, k) => ({ label: short(n), data: S.map((p) => p[k]), backgroundColor: col(IFACE_KIND[n]), borderColor: css("--surface"), borderWidth: 1.5, barPercentage: 0.62 })) },
      options: { plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw}%` } } }, scales: { x: { ...noGrid, stacked: true }, y: { ...pctAxis(100), stacked: true } } }
    });
  }
  function drawPareto() {
    const pts = D.pareto;
    const lab = { "Action-Centric-WAM": { label: "Action-Centric", dx: 0, dy: -16, align: "center" }, "Action-Only-WAM": { label: "Action-Only", dx: 0, dy: 16, align: "center" }, "Joint-WAM": { label: "Joint", dx: -12, dy: 0, align: "right" }, "IDM-WAM": { label: "IDM", dx: 0, dy: -16, align: "center" }, "UniW(A)M + PAIR": { label: "UniW(A)M + PAIR", bold: true, dx: 14 } };
    make("chart-pareto", {
      type: "scatter",
      data: { datasets: pts.map((p) => ({ label: p.name, data: [{ x: p.t, y: p.s, ...lab[p.name] }], backgroundColor: col(p.kind), borderColor: col(p.kind), pointStyle: p.kind === "ours" ? "rectRot" : "circle", pointRadius: p.kind === "ours" ? 11 : 8, pointHoverRadius: p.kind === "ours" ? 13 : 10, borderWidth: 0 })) },
      options: {
        layout: { padding: { right: 20, top: 10 } },
        plugins: { legend: { display: false }, pointLabels: { enabled: true }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw.y}% · ${c.raw.x} s` } } },
        scales: { x: { min: 220, max: 440, grid: { color: css("--grid") }, border: { display: false }, title: { display: true, text: "Average completion time (s)" } }, y: { ...pctAxis(85), min: 40, title: { display: true, text: "Success rate" } } }
      }
    });
  }
  function drawLatency() {
    const names = D.interfaces.slice().sort((a, b) => D.latency[a] - D.latency[b]);
    make("chart-latency", {
      type: "bar",
      data: { labels: names.map(short), datasets: [{ data: names.map((n) => D.latency[n]), backgroundColor: names.map((n) => col(IFACE_KIND[n])), borderRadius: radius(), barPercentage: 0.62 }] },
      options: { indexAxis: "y", layout: { padding: { right: 56 } }, plugins: { legend: { display: false }, valueLabels: { enabled: true, fmt: (v) => v.toFixed(1) + " ms" } }, scales: { x: { min: 0, max: 600, grid: { color: css("--grid") }, border: { display: false }, ticks: { callback: (v) => v + " ms" } }, y: noGrid } }
    });
  }
  function buildAdapt() {
    let h = "<thead><tr><th>Method</th><th>Before LoRA</th><th>After LoRA</th></tr></thead><tbody>";
    D.adaptation.forEach((r) => {
      const ours = r[0].startsWith("PAIR");
      h += `<tr class="${ours ? "ours" : ""}"><td>${r[0]}</td><td class="${ours ? "best" : ""}">${r[1].toFixed(1)}</td><td>${r[2] == null ? "—" : r[2].toFixed(1)}</td></tr>`;
    });
    $("#tbl-adapt").innerHTML = h + "</tbody>";
  }

  /* ---------- Ablations ---------- */
  function drawProg() {
    const P = D.progressive;
    make("chart-prog", {
      type: "line",
      data: { labels: P.steps, datasets: D.interfaces.map((n) => ({ label: short(n), data: P[n], borderColor: col(IFACE_KIND[n]), backgroundColor: col(IFACE_KIND[n]), borderWidth: 2.5, pointRadius: 5, pointHoverRadius: 7, tension: 0, spanGaps: false })) },
      options: { plugins: { legend: { position: "bottom" } }, scales: { y: { ...pctAxis(100), min: 40 }, x: { ...noGrid, ticks: { font: { size: 11 } } } } }
    });
  }
  function drawAbl() {
    const A = D.ablation; const colors = [css("--c-base2"), css("--c-base"), css("--accent-soft"), css("--accent")];
    make("chart-abl", {
      type: "bar",
      data: { labels: D.interfaces.map(short), datasets: A.configs.map((c, i) => ({ label: c, data: D.interfaces.map((n) => A.avg[n][i]), backgroundColor: colors[i], borderRadius: radius(), barPercentage: 0.86, categoryPercentage: 0.78 })) },
      options: { plugins: { legend: { position: "bottom" } }, scales: { y: pctAxis(100), x: noGrid } }
    });
  }
  function drawRemoval() {
    const R = D.removal;
    make("chart-removal", {
      type: "bar",
      data: {
        labels: R.map((r) => [short(r.iface), `${r.world[0]} / ${r.action[0]}`]),
        datasets: [
          { label: "Full model", data: R.map((r) => r.full), backgroundColor: css("--accent"), borderRadius: radius() },
          { label: "w/o world-prediction expert", data: R.map((r) => r.world[1]), backgroundColor: css("--c-base"), borderRadius: radius() },
          { label: "w/o action expert", data: R.map((r) => r.action[1]), backgroundColor: css("--c-base2"), borderRadius: radius() }
        ]
      },
      options: { plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => { const r = R[c.dataIndex]; const e = c.datasetIndex === 1 ? ` (${r.world[0]})` : c.datasetIndex === 2 ? ` (${r.action[0]})` : ""; return ` ${c.dataset.label}${e}: ${c.raw}%`; } } } }, scales: { y: pctAxis(100), x: { ...noGrid, ticks: { font: { size: 11 } } } } }
    });
  }
  function drawUtil() {
    const U = D.utilization;
    make("chart-util", {
      type: "bar",
      data: {
        labels: U.map((u) => `${short(u[0])} · ${u[1].toLowerCase()}`),
        datasets: [
          { label: "Shared expert", data: U.map((u) => u[3]), backgroundColor: css("--c-base2") },
          { label: "Atomic expert", data: U.map((u) => +(100 - u[3]).toFixed(1)), backgroundColor: U.map((u) => col(IFACE_KIND[u[0]])) }
        ]
      },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.datasetIndex ? U[c.dataIndex][2] + " (atomic)" : "Shared"}: ${c.raw}%` } } }, scales: { x: { ...pctAxis(100), stacked: true }, y: { ...noGrid, stacked: true, ticks: { font: { size: 11 } } } } }
    });
  }
  function buildDense() {
    let h = "<thead><tr><th>Architecture</th><th>Joint</th><th>IDM</th><th>Action-Centric</th><th>Action-Only</th><th>Average</th></tr></thead><tbody>";
    D.dense.forEach((r) => { const ours = r[0] === "UniW(A)M"; h += `<tr class="${ours ? "ours" : ""}"><td>${r[0]}</td>` + r.slice(1).map((v) => `<td class="${ours ? "best" : ""}">${v.toFixed(1)}</td>`).join("") + "</tr>"; });
    $("#tbl-dense").innerHTML = h + "</tbody>";
  }

  /* ---------- Interface explorer ---------- */
  const tok = (t) => t.replace(/^([fa])(\w)$/, "$1<sub>$2</sub>");
  function maskHTML(tokens, rows, M) {
    let h = `<div class="mgrid" style="grid-template-columns: 28px repeat(${tokens.length}, 1fr)"><div></div>`;
    tokens.forEach((t) => (h += `<div class="mh">${tok(t)}</div>`));
    rows.forEach((r, i) => {
      h += `<div class="mh">${tok(r)}</div>`;
      M[i].forEach((v) => (h += `<div class="mc ${v ? (r[0] === "f" ? "v" : "a") : "z"}"></div>`));
    });
    return h + "</div>";
  }
  function showIface(name) {
    const E = D.explorer[name];
    document.documentElement.style.setProperty("--ex-color", col(E.kind));
    $("#ex-role").textContent = E.role; $("#ex-name").textContent = name; $("#ex-desc").textContent = E.desc;
    const chips = (arr) => arr.map((c) => `<span class="chip ${c === "Shared" ? "chip-shared" : ""}">${c}</span>`).join("");
    $("#ex-caps").innerHTML = chips(E.caps); $("#ex-train").innerHTML = chips(E.train); $("#ex-infer").innerHTML = chips(E.infer);
    $("#ex-deploy").textContent = E.deploy;
    $("#mask-train").innerHTML = maskHTML(E.mask.trainTokens, E.mask.trainRows, E.mask.train);
    $("#mask-infer").innerHTML = maskHTML(E.mask.inferTokens, E.mask.inferRows, E.mask.infer);
  }

  /* ---------- Videos (list in videos.js) ---------- */
  const VID = window.UNIWAM_VIDEOS;
  const io = new IntersectionObserver((es) => es.forEach((e) => (e.isIntersecting ? e.target.play().catch(() => {}) : e.target.pause())), { threshold: 0.2 });
  const badge = (x) => (x && x > 1.15 ? `<span class="vbadge">${x}×</span>` : "");
  function clip(file, label, speed, cls = "", tag = "") {
    return `<figure class="vclip ${cls}"><div class="vwrap"><video data-src="assets/videos/${file}.mp4" poster="assets/videos/${file}.jpg" muted loop playsinline preload="none"></video>${badge(speed)}</div>
      <figcaption>${tag ? `<span class="vtag">${tag}</span>` : ""}${label}</figcaption></figure>`;
  }
  function buildVideos() {
    let h = "";
    Object.entries(VID.real).forEach(([k, t], i) => {
      h += `<div class="vid-pane ${i ? "" : "on"}" data-t="${k}">
        <div class="vgrid vgrid-3">${t.clips.map(([f, l, sp]) => clip(f, l, sp)).join("")}</div></div>`;
    });
    $("#real-video-panes").innerHTML = h;
    $("#robotwin-videos").innerHTML = VID.robotwin.map(([f, l]) => clip(f, l, 0)).join("");
    $("#libero-videos").innerHTML = VID.libero.map(([f, s, l]) => clip(f, l, 0, "vclip-sq", s)).join("");
    // load each video only when it comes near the viewport, then play while visible
    $$("video[data-src]").forEach((v) => {
      const near = new IntersectionObserver((es) => {
        if (!es[0].isIntersecting) return;
        v.src = v.dataset.src; delete v.dataset.src; near.disconnect(); io.observe(v);
      }, { rootMargin: "300px" });
      near.observe(v);
    });
  }

  /* ---------- wiring ---------- */
  function drawAll() {
    applyDefaults();
    drawReal(); drawRobotwin(); drawWA(); drawOOD(); drawSel(); drawPareto(); drawLatency(); drawProg(); drawAbl(); drawRemoval(); drawUtil();
    showIface($("#iface-tabs .on").textContent);
  }
  function init() {
    seg($("#iface-tabs"), Object.keys(D.explorer), showIface, 1);
    seg($("#rw-tabs"), ["Average", ...D.tasks.map((t) => t.replace("Tabletop Cleaning", "Tabletop"))], (_, i) => { rwIdx = i === 0 ? 3 : i - 1; drawReal(); });
    seg($("#ood-tabs"), ["Average", "Clothes", "Box", "Tabletop"], (_, i) => { oodIdx = i === 0 ? 3 : i - 1; drawOOD(); });
    seg($("#sel-tabs"), ["Clothes", "Box", "Tabletop"], (_, i) => { selTask = D.tasks[i]; drawSel(); });
    seg($("#wa-tabs"), ["Clothes", "Box", "Tabletop"], (_, i) => { waTask = D.tasks[i]; drawWA(); });
    bindStaticSeg($("#rt-tabs"), (b) => { rtKey = b.dataset.k; drawRobotwin(); });
    bindStaticSeg($("#vid-tabs"), (b) => $$(".vid-pane").forEach((p) => p.classList.toggle("on", p.dataset.t === b.dataset.t)));
    bindStaticSeg($("#seq-tabs"), (b) => ($("#seq-img").src = `assets/fig/${b.dataset.img}.svg`));
    bindStaticSeg($("#qual-tabs"), (b) => ($("#qual-img").src = `assets/fig/${b.dataset.img}.svg`));
    const hv = $(".hero-video video");
    if (hv) {
      // pause the teaser off-screen; resume only if it was playing (respects a manual pause)
      let wasPlaying = true;
      new IntersectionObserver((es) => {
        if (es[0].isIntersecting) { if (wasPlaying) hv.play().catch(() => {}); }
        else { wasPlaying = !hv.paused; hv.pause(); }
      }, { threshold: 0.2 }).observe(hv);
      const so = $(".sound-on");
      const sync = () => so && so.classList.toggle("gone", !hv.muted && hv.volume > 0);
      if (so) so.addEventListener("click", () => { hv.muted = false; if (hv.volume === 0) hv.volume = 0.8; hv.play().catch(() => {}); sync(); });
      hv.addEventListener("volumechange", sync);
    }
    buildLibero(); buildEWM(); buildAdapt(); buildDense(); buildVideos();
    if (HAS_CHART) drawAll();

    // copy bibtex
    $("#copy-bib").onclick = () => navigator.clipboard.writeText($("#bib-text").textContent).then(() => { $("#copy-bib").textContent = "Copied"; setTimeout(() => ($("#copy-bib").textContent = "Copy"), 1500); });

    // nav highlight
    const links = $$(".nav-links a");
    const so = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-45% 0px -50% 0px" });
    $$("main section[id]").forEach((s) => so.observe(s));
    window.addEventListener("scroll", () => document.body.classList.toggle("scrolled", scrollY > 10), { passive: true });

    // reveal on scroll
    const ro = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); ro.unobserve(e.target); } }), { threshold: 0.08 });
    $$(".sec-head, .panel, .figure, .card, .task, .explorer, .stat").forEach((el) => { el.classList.add("reveal"); ro.observe(el); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
