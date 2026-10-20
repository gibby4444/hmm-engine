// Browser port of core/gating.py, core/confidence_sizing.py, core/vol_integration.py
// and backtest/engine.py. Verified against Python reference metrics (web/verify.js).
const Engine = (() => {
  function gates(P, thr, persist, neutral) {
    const T = P.length, regime = new Int8Array(T), conf = new Float64Array(T);
    const am = (r) => { let k = 0; for (let i = 1; i < r.length; i++) if (r[i] > r[k]) k = i; return k; };
    let raw0 = am(P[0]);
    let confirmed = P[0][raw0] >= thr ? raw0 : (neutral ? -1 : raw0);
    let cand = -9, streak = 0, flickers = 0, switches = 0, prevRaw = raw0;
    for (let t = 0; t < T; t++) {
      const p = P[t], top = am(p), pTop = p[top];
      if (t > 0 && top !== prevRaw) flickers++;
      prevRaw = top;
      if (top !== confirmed && pTop >= thr) {
        streak = cand === top ? streak + 1 : 1;
        cand = top;
        if (streak >= persist) { confirmed = top; cand = -9; streak = 0; }
      } else {
        cand = -9; streak = 0;
        if (neutral && confirmed !== -1 && Math.max(...p) < thr && p[confirmed] < thr) confirmed = -1;
      }
      if (t > 0 && confirmed !== regime[t - 1]) switches++;
      regime[t] = confirmed;
      conf[t] = confirmed >= 0 ? p[confirmed] : Math.max(...p);
    }
    return { regime, conf, flickers, switches };
  }

  function multiplier(p, s, K) {
    const lo = 1 / K, thr = s.threshold, hi = s.full;
    if (s.mapping === "stepped") {
      if (p >= hi) return 1;
      if (p >= (thr + hi) / 2) return (1 + s.floor) / 2;
      if (p >= thr) return s.floor;
      return s.floor / 2;
    }
    if (p >= thr) return s.floor + (1 - s.floor) * Math.min(1, Math.max(0, (p - thr) / Math.max(1e-9, hi - thr)));
    return s.floor * Math.min(1, Math.max(0, (p - lo) / Math.max(1e-9, thr - lo)));
  }

  // exposure series aligned to the FULL date index (zeros before walk-forward start)
  function hmmExposure(A, o) {
    const P = A.P;
    // "confidence alone" = no gate, sizing still anchored at the threshold (mirrors gating_eval.sizing_modes)
    const g = o.mode === "confidence" ? gates(P, 0, 1, false) : gates(P, o.threshold, o.persistence, o.neutral);
    const N = A.close.length, out = new Float64Array(N), i0 = A.oos_start;
    const cap = 1.0;
    for (let t = 0; t < P.length; t++) {
      const s = g.regime[t];
      let e = s < 0 ? 0 : A.playbooks[A.version[t]][s];
      if (o.mode === "confidence" || o.mode === "both")
        e *= multiplier(g.conf[t], { threshold: o.threshold, floor: o.floor, full: o.full, mapping: o.mapping }, A.n_states);
      e = Math.max(-cap, Math.min(cap, e));
      if (o.breaker != null) {
        const v = A.ext[i0 + t];
        if (v != null && v > o.breaker) e = Math.min(e, 0);   // historical gaps: hold (as the harness does)
      }
      out[i0 + t] = e;
    }
    return { exp: out, g };
  }

  function maFilter(close, w) {
    const N = close.length, out = new Float64Array(N);
    let sum = 0;
    for (let i = 0; i < N; i++) {
      sum += close[i];
      if (i >= w) sum -= close[i - w];
      out[i] = i >= w - 1 ? (close[i] > sum / w ? 1 : 0) : 0;
    }
    return out;
  }

  function volTarget(close, target, lb, bpy) {
    const N = close.length, out = new Float64Array(N), r = new Float64Array(N);
    for (let i = 1; i < N; i++) r[i] = Math.log(close[i] / close[i - 1]);
    for (let i = 0; i < N; i++) {
      if (i < lb) { out[i] = 0; continue; }          // rolling(lb) on diff() needs lb returns (bars 1..lb)
      let m = 0; for (let j = i - lb + 1; j <= i; j++) m += r[j]; m /= lb;
      let v = 0; for (let j = i - lb + 1; j <= i; j++) v += (r[j] - m) ** 2;
      const sd = Math.sqrt(v / (lb - 1)) * Math.sqrt(bpy);
      out[i] = sd > 0 ? Math.min(1, target / sd) : 0;
    }
    return out;
  }

  function backtest(close, exp, start, costBps, bpy) {
    const N = close.length, n = N - start;
    const ret = new Float64Array(n), eq = new Float64Array(n), dd = new Float64Array(n);
    let e = 1, peak = 1, trades = 0, cost = 0, expSum = 0, mdd = 0;
    for (let k = 0; k < n; k++) {
      const i = start + k;
      const pos = exp[i], prev = k === 0 ? 0 : exp[i - 1];
      const turn = Math.abs(pos - prev);
      if (k > 0 && turn > 1e-9) trades++;
      const nxt = i + 1 < N ? close[i + 1] / close[i] - 1 : 0;
      const c = turn * costBps / 1e4;
      cost += c;
      ret[k] = pos * nxt - c;
      e *= 1 + ret[k]; eq[k] = e;
      if (e > peak) peak = e;
      dd[k] = e / peak - 1; if (dd[k] < mdd) mdd = dd[k];
      expSum += Math.abs(pos);
    }
    let mu = 0; for (const x of ret) mu += x; mu /= n;
    let v = 0, dv = 0, nd = 0, dm = 0;
    for (const x of ret) v += (x - mu) ** 2;
    const sd = Math.sqrt(v / (n - 1));
    for (const x of ret) if (x < 0) { dm += x; nd++; }
    dm /= Math.max(1, nd);
    for (const x of ret) if (x < 0) dv += (x - dm) ** 2;
    const dsd = Math.sqrt(dv / Math.max(1, nd - 1)) * Math.sqrt(bpy);
    const yrs = n / bpy;
    const cagr = e > 0 ? Math.pow(e, 1 / yrs) - 1 : -1;
    return {
      eq, dd, cagr, vol: sd * Math.sqrt(bpy), sharpe: sd > 0 ? mu / sd * Math.sqrt(bpy) : 0,
      sortino: dsd > 0 ? mu * bpy / dsd : NaN, max_dd: mdd, calmar: mdd < 0 ? cagr / -mdd : NaN,
      trades, exposure: expSum / n, cost_drag: cost / yrs,
    };
  }

  function run(A, o) {
    const h = hmmExposure(A, o), i0 = A.oos_start;
    const ma = maFilter(A.close, A.ma_window);
    const vt = volTarget(A.close, A.vt_target, A.vt_lookback, A.bpy);
    const bh = new Float64Array(A.close.length).fill(1);
    const cost = o.costs;
    const res = {
      hmm: backtest(A.close, h.exp, i0, cost, A.bpy),
      ma: backtest(A.close, ma, i0, cost, A.bpy),
      vt: backtest(A.close, vt, i0, cost, A.bpy),
      bh: backtest(A.close, bh, i0, cost, A.bpy),
    };
    res.single = A.single === "vol_target" ? "vt" : "bh";
    res.gate = h.g; res.exp = h.exp;
    let maSw = 0; for (let i = i0 + 1; i < ma.length; i++) if (ma[i] !== ma[i - 1]) maSw++;
    res.maSwitches = maSw;
    res.pass = res.hmm.sharpe > res.ma.sharpe && res.hmm.sharpe > res[res.single].sharpe;
    return res;
  }

  function prepare(A) {
    A.P = A.probs.map((r) => r.map((x) => x / 10000));
    return A;
  }

  return { gates, multiplier, hmmExposure, maFilter, volTarget, backtest, run, prepare };
})();
if (typeof module !== "undefined") module.exports = Engine;
