(function () {
  "use strict";
  const K = window.KLogic;
  const $ = (id) => document.getElementById(id);
  const fetchJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + " (" + r.status + ")"); return r.json(); });
  const fmt = (v, nd) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(nd === undefined ? 1 : nd));
  const pct = (p) => (p === null || p === undefined ? "–" : (p * 100).toFixed(p < 0.1 ? 1 : 0) + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let meta, grid, map, rects = [], obsLayer;
  const state = { di: 0, prov: -1, sel: null, day: null, feat: null, pArr: [], timer: null, token: 0, metricsLoaded: false };
  const cache = {}, fcache = {};

  init().catch((err) => {
    console.error(err);
    document.body.insertAdjacentHTML("beforeend", '<div class="card" style="margin:12px">Gagal memuat data: ' + esc(err.message) + "</div>");
  });

  async function init() {
    [meta, grid] = await Promise.all([fetchJSON("data/dates.json"), fetchJSON("data/grid.json")]);
    $("demo-banner").hidden = !meta.demo;
    renderInfo();
    $("labeldef").textContent = meta.label_def;
    $("generated").textContent = "Data dibuat: " + meta.generated + (meta.demo ? " (DEMO)" : "");
    setupTabs(); setupMap(); setupControls(); renderLegend();
    await goTo(meta.dates.length - 1);
    window.__klab = { select, goTo, state }; // kait kecil untuk uji otomatis
  }

  // ---------------------------------------------------------------- peta
  function setupMap() {
    map = L.map("map", { preferCanvas: true });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 12, subdomains: "abcd", attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    }).addTo(map);
    const canvas = L.canvas({ padding: 0.5 });
    const h = grid.cell_deg / 2;
    let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
    for (let i = 0; i < grid.n; i++) {
      const la = grid.lat[i], lo = grid.lon[i];
      minLat = Math.min(minLat, la - h); maxLat = Math.max(maxLat, la + h);
      minLon = Math.min(minLon, lo - h); maxLon = Math.max(maxLon, lo + h);
      const r = L.rectangle([[la - h, lo - h], [la + h, lo + h]], { renderer: canvas, stroke: false, fill: true, fillColor: "#9aa5b1", fillOpacity: 0.7 });
      r.on("click", () => select(i));
      r.addTo(map);
      rects.push(r);
    }
    map.fitBounds([[minLat, minLon], [maxLat, maxLon]]);
    obsLayer = L.layerGroup();
  }

  function renderLegend() {
    const rows = meta.levels.map((l) => '<div><i style="background:' + l.color + '"></i>' + esc(l.name) + " (≥" + Math.round(l.min * 100) + "%)</div>");
    rows.push('<div><i style="background:#9aa5b1"></i>Tidak ada data</div>');
    $("legend").innerHTML = rows.join("");
  }

  // ---------------------------------------------------------------- kontrol
  function setupTabs() {
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".tabpane").forEach((p) => (p.hidden = p.id !== "tab-" + b.dataset.tab));
      if (b.dataset.tab === "map") setTimeout(() => map.invalidateSize(), 0);
      if (b.dataset.tab === "metrics") loadMetrics();
    }));
  }

  function setupControls() {
    const sel = $("prov");
    sel.innerHTML = '<option value="-1">Semua provinsi</option>' + grid.provinces.map((p, i) => '<option value="' + i + '">' + esc(p) + "</option>").join("");
    sel.addEventListener("change", () => { state.prov = Number(sel.value); paint(); });
    const sl = $("slider");
    sl.max = meta.dates.length - 1;
    sl.addEventListener("input", () => { stop(); goTo(Number(sl.value)); });
    $("play").addEventListener("click", () => (state.timer ? stop() : play()));
    $("obs").disabled = !meta.has_obs;
    $("obs").addEventListener("change", paint);
  }

  function play() {
    if (state.di >= meta.dates.length - 1) state.di = -1;
    $("play").textContent = "⏸";
    state.timer = setInterval(async () => {
      if (state.di >= meta.dates.length - 1) return stop();
      await goTo(state.di + 1);
    }, 700);
  }
  function stop() { if (state.timer) clearInterval(state.timer); state.timer = null; $("play").textContent = "▶"; }

  async function goTo(di) {
    const token = ++state.token;
    const date = meta.dates[di];
    if (!cache[date]) cache[date] = await fetchJSON("data/d/" + date + ".json");
    if (state.sel !== null && !fcache[date]) fcache[date] = await fetchJSON("data/f/" + date + ".json");
    if (token !== state.token) return; // ada permintaan yang lebih baru
    state.di = di; state.day = cache[date]; state.feat = fcache[date] || null;
    state.pArr = state.day.p.map((v) => (v === null ? null : v / 1000));
    $("slider").value = di;
    paint();
  }

  // ---------------------------------------------------------------- gambar ulang
  function paint() {
    const date = meta.dates[state.di];
    const sp = meta.splits && meta.splits[state.di];
    $("datelabel").textContent = "Dibuat " + date + " · berlaku " + K.addDays(date, 1) + " s/d " + K.addDays(date, meta.horizon) + (sp ? " · data " + sp : "");
    for (let i = 0; i < grid.n; i++) {
      const li = K.levelIndex(state.pArr[i], meta.levels);
      const inProv = state.prov < 0 || grid.prov[i] === state.prov;
      rects[i].setStyle({ fillColor: li < 0 ? "#9aa5b1" : meta.levels[li].color, fillOpacity: inProv ? 0.72 : 0.1 });
    }
    obsLayer.clearLayers();
    const showObs = $("obs").checked && state.day.y;
    if (showObs) {
      for (const i of state.day.y) {
        if (state.prov >= 0 && grid.prov[i] !== state.prov) continue;
        L.circleMarker([grid.lat[i], grid.lon[i]], { radius: 3, color: "#111", weight: 1, fillColor: "#fff", fillOpacity: 1, interactive: false }).addTo(obsLayer);
      }
      obsLayer.addTo(map);
    } else map.removeLayer(obsLayer);
    renderSummary(); renderSeries();
    if (state.sel !== null) renderCell(false);
  }

  function renderSummary() {
    const s = K.summarize(state.pArr, grid.prov, state.prov, meta.levels);
    let h = '<div class="muted">' + s.n + " sel · rata-rata skor " + pct(s.mean) + "</div>";
    meta.levels.forEach((l, k) => {
      const w = s.n ? (100 * s.counts[k]) / s.n : 0;
      h += '<div class="bar"><span class="nm">' + esc(l.name) + '</span><span class="b"><span style="width:' + w + "%;background:" + l.color + '"></span></span><span class="v">' + s.counts[k] + " (" + w.toFixed(0) + "%)</span></div>";
    });
    $("summary").innerHTML = h;
  }

  function renderSeries() {
    const c = $("ts"), ctx = c.getContext("2d"), W = c.width, H = c.height, m = { l: 30, r: 6, t: 14, b: 16 };
    const key = state.prov < 0 ? "ALL" : grid.provinces[state.prov], t = meta.ts[key];
    const fg = getComputedStyle(document.body).color;
    ctx.clearRect(0, 0, W, H); ctx.font = "10px system-ui"; ctx.fillStyle = fg; ctx.strokeStyle = fg;
    const obs = t.obs.filter((v) => v !== null);
    const ymax = Math.max(0.05, ...t.mean, ...t.high, ...obs);
    const X = (i) => m.l + (t.mean.length > 1 ? (i / (t.mean.length - 1)) * (W - m.l - m.r) : 0);
    const Y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(m.l, H - m.b); ctx.lineTo(W - m.r, H - m.b); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillText(Math.round(ymax * 100) + "%", 2, m.t + 4); ctx.fillText("0%", 12, H - m.b);
    const line = (arr, color, dash) => {
      ctx.strokeStyle = color; ctx.setLineDash(dash || []); ctx.lineWidth = 1.5; ctx.beginPath();
      let started = false;
      arr.forEach((v, i) => { if (v === null) return; started ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); started = true; });
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(t.mean, "#1f6feb"); line(t.high, "#d7301f");
    if (meta.has_obs) line(t.obs, "#8a8f98", [4, 3]);
    ctx.strokeStyle = fg; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(X(state.di), m.t); ctx.lineTo(X(state.di), H - m.b); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = "#1f6feb"; ctx.fillText("rata-rata", m.l, 9);
    ctx.fillStyle = "#d7301f"; ctx.fillText("% sel Tinggi+", m.l + 50, 9);
    if (meta.has_obs) { ctx.fillStyle = "#8a8f98"; ctx.fillText("% aktual", m.l + 120, 9); }
  }

  // ---------------------------------------------------------------- sel terpilih + what-if
  async function select(i) {
    state.sel = i;
    const date = meta.dates[state.di];
    if (!fcache[date]) fcache[date] = await fetchJSON("data/f/" + date + ".json");
    state.feat = fcache[date];
    renderCell(true);
  }

  function featureValue(key, i) {
    if (key === "peat_frac") return grid.static && grid.static.peat_frac ? grid.static.peat_frac[i] : null;
    return state.feat && state.feat[key] ? state.feat[key][i] : null;
  }

  function renderCell(rebuildSliders) {
    const i = state.sel, p = state.pArr[i], li = K.levelIndex(p, meta.levels);
    const L_ = li >= 0 ? meta.levels[li] : null;
    const obsSet = state.day.y ? new Set(state.day.y) : null;
    let h = '<div class="muted">Sel ' + grid.ids[i] + " · " + esc(grid.provinces[grid.prov[i]]) + " · " + fmt(grid.lat[i], 2) + ", " + fmt(grid.lon[i], 2) + "</div>";
    h += '<div class="big">' + pct(p) + ' <span class="badge" style="background:' + (L_ ? L_.color : "#9aa5b1") + '">' + (L_ ? esc(L_.name) : "Tidak ada data") + '</span></div><div class="muted" style="font-size:11px">skor model, belum dikalibrasi</div>';
    if (L_ && L_.advice) h += '<div class="muted" style="margin:4px 0">' + esc(L_.advice) + "</div>";
    if (obsSet) h += "<div>Aktual: " + (obsSet.has(i) ? "<b>ada hotspot</b>" : "tidak ada hotspot") + " dalam " + meta.horizon + " hari</div>";
    h += '<table style="margin-top:6px">';
    for (const f of meta.features) h += "<tr><td>" + esc(f.label) + "</td><td>" + fmt(featureValue(f.key, i), 2) + " " + esc(f.unit) + "</td></tr>";
    h += "</table>";
    const wf = meta.features.filter((f) => f.whatif && featureValue(f.key, i) !== null);
    if (wf.length && p !== null) {
      h += '<h2 style="margin-top:10px">Bagaimana jika…</h2><div id="wi">';
      for (const f of wf) {
        const v = featureValue(f.key, i), lo = Math.min(f.lo, v), hi = Math.max(f.hi, v), step = (hi - lo) / 100 || 0.01;
        h += '<div class="sl"><div class="top"><span>' + esc(f.label) + '</span><span id="wv-' + f.key + '">' + fmt(v, 1) + " " + esc(f.unit) + '</span></div><input type="range" data-k="' + f.key + '" min="' + lo + '" max="' + hi + '" step="' + step + '" value="' + v + '"></div>';
      }
      h += '<div id="wres"></div><button class="small" id="wreset">Atur ulang</button></div>';
    }
    $("cell").innerHTML = h;
    if (wf.length && p !== null) wireWhatIf(wf, i, p);
  }

  function wireWhatIf(wf, i, p) {
    const base = {}; wf.forEach((f) => (base[f.key] = featureValue(f.key, i)));
    const update = () => {
      const nv = {};
      document.querySelectorAll("#wi input[data-k]").forEach((el) => {
        nv[el.dataset.k] = Number(el.value);
        const f = wf.find((x) => x.key === el.dataset.k);
        $("wv-" + el.dataset.k).textContent = fmt(el.value, 1) + " " + f.unit;
      });
      const q = K.whatIf(p, base, nv, meta.curves), lv = meta.levels[K.levelIndex(q, meta.levels)];
      const d = (q - p) * 100;
      $("wres").innerHTML = 'Skor simulasi: <b>' + pct(q) + '</b> <span class="badge" style="background:' + lv.color + '">' + esc(lv.name) + "</span> <span class=\"muted\">(" + (d >= 0 ? "+" : "") + d.toFixed(1) + " poin)</span>";
    };
    document.querySelectorAll("#wi input[data-k]").forEach((el) => el.addEventListener("input", update));
    $("wreset").addEventListener("click", () => { document.querySelectorAll("#wi input[data-k]").forEach((el) => (el.value = base[el.dataset.k])); update(); });
    update();
  }

  // ---------------------------------------------------------------- metrik model
  async function loadMetrics() {
    if (state.metricsLoaded) return;
    try {
      const m = await fetchJSON("data/metrics.json");
      const rows = K.flatten(m).map(([k, v]) => "<tr><td>" + esc(k) + "</td><td>" + esc(typeof v === "number" ? v.toFixed(4) : v) + "</td></tr>");
      $("metrics").innerHTML = "<table>" + rows.join("") + "</table>";
    } catch (e) {
      $("metrics").textContent = "Belum ada metrik model. Taruh results/metrics.json (dari Orang B) lalu bangun ulang situs.";
    }
    $("figs").innerHTML = (meta.figures || []).map((f) => '<h2 style="margin-top:12px">' + esc(f.title) + '</h2><img src="' + esc(f.file) + '" alt="' + esc(f.title) + '" style="max-width:100%">').join("");
    state.metricsLoaded = true;
  }

  // ---------------------------------------------------------------- info model
  function renderInfo() {
    const mi = meta.model_info || {};
    if (mi.status && !meta.demo) {
      $("status-note").hidden = false;
      $("status-note").textContent = "Status model: " + mi.status + ". Skor belum dikalibrasi dan bukan peringatan resmi.";
    }
    let h = "<table>";
    [["Model", mi.model], ["Status", mi.status], ["Ambang F1 (batas Tinggi)", mi.threshold_f1 != null ? pct(mi.threshold_f1) : null],
     ["Ambang recall 70% (batas Sedang)", mi.threshold_recall70 != null ? pct(mi.threshold_recall70) : null],
     ["PR-AUC validasi", mi.validation_pr_auc != null ? Number(mi.validation_pr_auc).toFixed(3) : null]]
      .forEach(([k, v]) => { if (v) h += "<tr><td>" + esc(k) + "</td><td>" + esc(v) + "</td></tr>"; });
    h += "</table>";
    if ((mi.limitations || []).length) h += '<ul class="muted" style="margin:6px 0 0 16px;padding:0">' + mi.limitations.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>";
    $("modelinfo").innerHTML = h;
    if (meta.level_stats) {
      let t = '<div class="muted">Dari tanggal di luar data pelatihan. Laju dasar hotspot: ' + pct(meta.base_rate) + "</div><table><tr><th>Tingkat</th><th>Sel-hari</th><th>Ada hotspot dalam 7 hari</th></tr>";
      meta.level_stats.forEach((s) => { t += "<tr><td>" + esc(s.name) + "</td><td>" + s.n + "</td><td>" + (s.hit === null ? "–" : pct(s.hit)) + "</td></tr>"; });
      $("levelstats").innerHTML = t + "</table>";
    } else $("levelstats").textContent = "Tidak tersedia (tanggal tanpa label atau hanya data pelatihan).";
  }
})();
