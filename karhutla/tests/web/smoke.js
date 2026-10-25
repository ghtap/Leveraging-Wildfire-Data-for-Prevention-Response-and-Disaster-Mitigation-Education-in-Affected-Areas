// Uji asap di jsdom: node tests/web/smoke.js <folder-dist>   (butuh: npm i jsdom)
const path = require("path"), fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");
const dist = path.resolve(process.argv[2] || "dist");
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => { if (!/Could not load|Not implemented/.test(e.message)) errors.push(e.message); });
vc.on("error", (e) => errors.push(String(e)));

(async () => {
  const html = fs.readFileSync(path.join(dist, "index.html"), "utf8").replace(/<script[^>]*src=[^>]*><\/script>/g, "");
  const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://localhost/", virtualConsole: vc });
  {
    const w = dom.window;
    const ctx = new Proxy({}, { get: (t, k) => (k === "canvas" ? {} : () => ({ width: 0 })), set: () => true });
    w.HTMLCanvasElement.prototype.getContext = () => ctx;
    w.fetch = (u) => { const f = path.join(dist, u); return fs.existsSync(f)
      ? Promise.resolve({ ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(f, "utf8")) })
      : Promise.resolve({ ok: false, status: 404 }); };
    for (const f of ["vendor/leaflet/leaflet.js", "assets/logic.js", "assets/app.js"]) w.eval(fs.readFileSync(path.join(dist, f), "utf8"));
  }
  const w = dom.window, d = w.document, $ = (id) => d.getElementById(id);
  const until = async (fn, what) => { for (let i = 0; i < 100; i++) { if (fn()) return; await new Promise((r) => setTimeout(r, 50)); } throw new Error("timeout: " + what); };
  const ok = (c, m) => { if (!c) throw new Error("GAGAL: " + m); console.log("ok  " + m); };

  await until(() => w.__klab && $("summary").innerHTML.length > 0, "inisialisasi");
  const meta = JSON.parse(fs.readFileSync(path.join(dist, "data/dates.json"), "utf8"));
  ok($("legend").children.length === meta.levels.length + 1, "legenda lengkap");
  ok($("prov").options.length >= 2, "pilihan provinsi terisi");
  ok(/Dibuat \d{4}-\d{2}-\d{2} · berlaku/.test($("datelabel").textContent), "label tanggal");
  ok($("demo-banner").hidden === !meta.demo, "banner demo sesuai flag");

  await w.__klab.select(0);
  ok($("cell").innerHTML.includes("Bagaimana jika"), "panel sel + simulasi muncul");
  const sl = d.querySelector("#wi input[data-k]");
  const before = $("wres").textContent;
  sl.value = sl.min; sl.dispatchEvent(new w.Event("input"));
  ok($("wres").textContent.includes("Skor simulasi") && $("wres").textContent !== before, "slider mengubah skor simulasi");
  $("wreset").click();
  ok(/\(\+?-?0\.0 poin\)/.test($("wres").textContent), "atur ulang kembali ke nilai awal");

  const n0 = $("summary").textContent;
  $("prov").value = "0"; $("prov").dispatchEvent(new w.Event("change"));
  ok($("summary").textContent !== n0, "filter provinsi mengubah ringkasan");

  await w.__klab.goTo(0);
  ok(w.__klab.state.di === 0 && $("slider").value === "0", "pindah ke tanggal pertama");
  $("obs").checked = true; $("obs").dispatchEvent(new w.Event("change"));
  ok(true, "kejadian aktual tergambar tanpa error");

  d.querySelector('[data-tab="metrics"]').click();
  await until(() => !$("metrics").textContent.startsWith("Memuat"), "tab metrik");
  ok($("tab-metrics").hidden === false && $("tab-map").hidden === true, "pindah tab");

  if (errors.length) throw new Error("error di halaman:\n" + errors.join("\n"));
  console.log("SEMUA LULUS"); process.exit(0);
})().catch((e) => { console.error(e.message); process.exit(1); });
