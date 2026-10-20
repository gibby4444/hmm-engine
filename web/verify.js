// node web/verify.js — checks the browser engine against Python reference numbers
const E = require("./engine.js");
const d = require("./data.json");
let bad = 0;
for (const [sym, A] of Object.entries(d.assets)) {
  E.prepare(A);
  for (const r of A.refs) {
    const o = { threshold: r.threshold, persistence: r.persistence, neutral: A.gate.neutral && r.threshold > 0,
                mode: r.mode, floor: A.sizing.floor, full: A.sizing.full, mapping: A.sizing.mapping,
                breaker: r.breaker, costs: A.costs_bps };
    if (r.threshold === 0) o.neutral = false;
    const res = E.run(A, o);
    const h = res.hmm;
    const ok = Math.abs(h.sharpe - r.sharpe) < 2e-3 && Math.abs(h.cagr - r.cagr) < 2e-3 &&
               Math.abs(h.max_dd - r.max_dd) < 2e-3 && res.gate.switches === r.switches;
    if (!ok) bad++;
    console.log(sym, r.threshold, r.persistence, r.mode, r.breaker, ok ? "OK" : "MISMATCH",
      `sharpe ${h.sharpe.toFixed(4)}/${r.sharpe.toFixed(4)} cagr ${h.cagr.toFixed(4)}/${r.cagr.toFixed(4)}`,
      `dd ${h.max_dd.toFixed(4)}/${r.max_dd.toFixed(4)} sw ${res.gate.switches}/${r.switches}`);
  }
  const res = E.run(A, { threshold: A.gate.threshold, persistence: A.gate.persistence, neutral: A.gate.neutral,
    mode: "hard", floor: .5, full: .95, mapping: "linear", breaker: null, costs: A.costs_bps });
  console.log(sym, "baselines MA", res.ma.sharpe.toFixed(3), res.ma.max_dd.toFixed(3), "VT", res.vt.sharpe.toFixed(3),
    "BH", res.bh.sharpe.toFixed(3), res.bh.cagr.toFixed(3), "pass", res.pass);
}
process.exit(bad ? 1 : 0);
