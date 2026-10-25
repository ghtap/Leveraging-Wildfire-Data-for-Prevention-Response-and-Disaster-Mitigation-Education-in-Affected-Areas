/* Logika murni (tanpa DOM) agar bisa diuji di Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.KLogic = factory();
})(this, function () {
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const logit = (p) => { p = clamp(p, 1e-3, 1 - 1e-3); return Math.log(p / (1 - p)); };
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function levelIndex(p, levels) {
    if (p === null || p === undefined || Number.isNaN(p)) return -1;
    let idx = 0;
    for (let k = 0; k < levels.length; k++) if (p >= levels[k].min) idx = k;
    return idx;
  }

  // interpolasi linear sepotong-sepotong; di luar rentang ditahan di ujung
  function interp(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    for (let i = 1; i < xs.length; i++) {
      if (x <= xs[i]) {
        const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[i - 1] + t * (ys[i] - ys[i - 1]);
      }
    }
  }

  // newVals/baseVals: {fitur: nilai}. Efek tiap fitur dijumlahkan di skala logit (aproksimasi aditif).
  function whatIf(baseP, baseVals, newVals, curves) {
    let z = logit(baseP);
    for (const k of Object.keys(newVals)) {
      const c = curves[k];
      if (!c || baseVals[k] === null || baseVals[k] === undefined || newVals[k] === null) continue;
      const dz = (newVals[k] - baseVals[k]) / c.sigma;
      z += interp(c.z, c.dlogit, dz) - interp(c.z, c.dlogit, 0);
    }
    return sigmoid(z);
  }

  // p: array permil/1000 atau null; provIdx: indeks provinsi per sel; sel: -1 = semua
  function summarize(p, provIdx, sel, levels) {
    const counts = new Array(levels.length).fill(0);
    let n = 0, sum = 0;
    for (let i = 0; i < p.length; i++) {
      if (sel >= 0 && provIdx[i] !== sel) continue;
      if (p[i] === null || p[i] === undefined) continue;
      counts[levelIndex(p[i], levels)]++; n++; sum += p[i];
    }
    return { counts, n, mean: n ? sum / n : null };
  }

  function addDays(iso, k) {
    const d = new Date(iso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + k);
    return d.toISOString().slice(0, 10);
  }

  function flatten(obj, prefix = "", out = []) {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? prefix + "." + k : k;
      if (Array.isArray(v) && v.length && v.every((x) => x && typeof x === "object")) v.forEach((x, i) => flatten(x, key + "." + i, out));
      else if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, key, out);
      else out.push([key, Array.isArray(v) ? JSON.stringify(v) : v]);
    }
    return out;
  }

  return { clamp, logit, sigmoid, levelIndex, interp, whatIf, summarize, addDays, flatten };
});
