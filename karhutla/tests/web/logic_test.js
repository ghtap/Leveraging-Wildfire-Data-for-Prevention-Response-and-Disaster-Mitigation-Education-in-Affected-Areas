const K = require("../../src/app/web/assets/logic.js");
const assert = require("assert");
const levels = [{ min: 0 }, { min: 0.2 }, { min: 0.4 }, { min: 0.6 }];
assert.deepStrictEqual([0, 0.19, 0.2, 0.599, 0.6, 1].map((p) => K.levelIndex(p, levels)), [0, 0, 1, 2, 3, 3]);
assert.strictEqual(K.levelIndex(null, levels), -1);
assert.strictEqual(K.interp([-1, 0, 1], [0, 5, 10], -9), 0);            // ditahan di ujung
assert.strictEqual(K.interp([-1, 0, 1], [0, 5, 10], 0.5), 7.5);
const curves = { x: { sigma: 2, z: [-2, -1, 0, 1, 2], dlogit: [-1, -0.5, 0, 0.5, 1] } };
assert.ok(Math.abs(K.whatIf(0.3, { x: 5 }, { x: 5 }, curves) - 0.3) < 1e-9);   // tanpa perubahan = nilai awal
assert.ok(K.whatIf(0.3, { x: 5 }, { x: 7 }, curves) > 0.3);                    // naik 1 sigma -> risiko naik
assert.ok(K.whatIf(0.3, { x: 5 }, { x: 3 }, curves) < 0.3);
assert.ok(Math.abs(K.whatIf(0.3, { x: 5 }, { x: 105 }, curves) - K.whatIf(0.3, { x: 5 }, { x: 9 }, curves)) < 1e-9); // klem di 2 sigma
assert.ok(Math.abs(K.whatIf(0.3, { x: null }, { x: 9 }, curves) - 0.3) < 1e-9); // nilai dasar kosong diabaikan
const s = K.summarize([0.1, 0.5, null, 0.9], [0, 0, 1, 1], 1, levels.map((l) => l));
assert.deepStrictEqual([s.n, s.counts], [1, [0, 0, 0, 1]]);
assert.strictEqual(K.addDays("2026-10-30", 3), "2026-11-02");
assert.deepStrictEqual(K.flatten({ a: { b: 1 }, c: 2 }), [["a.b", 1], ["c", 2]]);
assert.deepStrictEqual(K.flatten({ r: [{ p: 1 }, { p: 2 }], f: ["x", "y"] }), [["r.0.p", 1], ["r.1.p", 2], ["f", "[\"x\",\"y\"]"]]);
console.log("logic.js OK");
