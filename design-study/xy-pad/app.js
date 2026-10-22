(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const LT = JSON.parse(document.getElementById("latent").textContent);
const L = D.labels, NL = L.length;
const KI = Object.fromEntries(D.knobs.map((k, i) => [k.name, i]));
const NK = D.knobs.length;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let COL = [];
const readColors = () => { COL = [...L.map((_, i) => cssVar(`--c${i}`)), cssVar("--c8")]; };
readColors();
const store = { get(k) { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };

// ---------- knob <-> group math (knob pad) ----------
const unitOf = (ki, code) => { const k = D.knobs[ki], s = k.p95 - k.p5 || 1e-6; return Math.min(1, Math.max(0, (code - k.p5) / s)); };
const codeOf = (ki, u) => { const k = D.knobs[ki]; return k.p5 + u * (k.p95 - k.p5); };
function groupPos(g, vals) {
  let s = 0, n = 0;
  for (const name of g.knobs) { const i = KI[name], v = vals[i]; if (v == null || Number.isNaN(v)) continue;
    const u = unitOf(i, v); s += g.signs[name] > 0 ? u : 1 - u; n++; }
  return n ? s / n : null;
}
function setGroup(g, u, vals) { for (const name of g.knobs) { const i = KI[name]; vals[i] = codeOf(i, g.signs[name] > 0 ? u : 1 - u); } }
const nat = (name, vals) => { const i = KI[name], v = vals[i]; if (v == null || Number.isNaN(v)) return null; const [lo, hi] = D.knobs[i].natural_range; return lo + v * (hi - lo); };

// ---------- category forest ----------
const T = D.forest.trees;
function predict(vals) {
  const p = new Float64Array(NL);
  for (const t of T) { let n = 0; while (t.l[n] !== -1) { const x = vals[t.f[n]]; n = ((x == null || Number.isNaN(x)) ? -1 : x) <= t.t[n] ? t.l[n] : t.r[n]; }
    const v = t.v[n]; for (let c = 0; c < NL; c++) p[c] += v[c]; }
  for (let c = 0; c < NL; c++) p[c] /= T.length;
  return p;
}
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

// ---------- presets and knob-pad starting points ----------
const PV = D.presets.values, PL = D.presets.label;
function baseFor(label) {
  const rows = PV.filter((_, i) => label < 0 || PL[i] === label);
  const out = new Array(NK).fill(NaN);
  for (let k = 0; k < NK; k++) { const xs = rows.map(r => r[k]).filter(v => v != null).sort((a, b) => a - b);
    if (xs.length >= rows.length * 0.5) out[k] = xs[Math.floor(xs.length / 2)]; }
  return out;
}
const BASES = [{ name: "Typical preset (all categories)", vals: baseFor(-1) }, ...L.map((l, i) => ({ name: `Typical ${l}`, vals: baseFor(i) }))];

// ---------- latent model: float16 weights -> Float32Array, small MLPs ----------
function f16(b64) {
  const bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}
const lay = (l) => ({ o: l.shape[0], i: l.shape[1], w: f16(l.w), b: f16(l.b) });
const DP = LT.dec_p.map(lay), DA = LT.dec_a.map(lay);
const ZD = LT.z_dim, NC = LT.n_cls;
const PCA = { C: f16(LT.pca.components), m: f16(LT.pca.mean), s: LT.pca.scale, T: LT.pca.frames };
const ACT = { w: f16(LT.activity.w), b: f16(LT.activity.b) };
const NB = LT.bools.length, CAT_SIZES = LT.cats.map(c => c.options.length);
const silu = (x) => x / (1 + Math.exp(-x));
function linear(l, x, out, act) {
  for (let o = 0; o < l.o; o++) { let s = l.b[o]; const r = o * l.i; for (let i = 0; i < l.i; i++) s += l.w[r + i] * x[i]; out[o] = act ? silu(s) : s; }
  return out;
}
function mlp(layers, x) {
  let h = x;
  layers.forEach((l, k) => { h = linear(l, h, new Float32Array(l.o), k < layers.length - 1); });
  return h;
}
function zin(z, y) { const x = new Float32Array(ZD + NC); x.set(z); x[ZD + y] = 1; return x; }
// decoder outputs: [page knobs (sigmoid) | display on/offs (logit) | oscillator type logits]
function decodeP(z, y) {
  const x = zin(z, y), out = mlp(DP, x);
  const num = new Float32Array(NK), act = new Uint8Array(NK);
  for (let k = 0; k < NK; k++) {
    num[k] = 1 / (1 + Math.exp(-out[k]));
    let s = ACT.b[k]; const r = k * (ZD + NC); for (let i = 0; i < ZD + NC; i++) s += ACT.w[r + i] * x[i];
    act[k] = s > 0 ? 1 : 0;
  }
  const bools = new Uint8Array(NB); for (let j = 0; j < NB; j++) bools[j] = out[NK + j] > 0 ? 1 : 0;
  const cats = []; let off = NK + NB;
  for (const n of CAT_SIZES) { let b = 0; for (let i = 1; i < n; i++) if (out[off + i] > out[off + b]) b = i; cats.push(b); off += n; }
  return { num, act, bools, cats };
}
// sound decoder -> 48-number sound code -> approximate mel (frames x 64), peak-relative, floored at -80 dB
const NM = LT.band_hz.length;
function decodeA(z, y) {
  const c = mlp(DA, zin(z, y)), F = PCA.T * NM, X = new Float32Array(F);
  for (let j = 0; j < F; j++) X[j] = PCA.m[j];
  for (let k = 0; k < c.length; k++) { const a = c[k] * PCA.s, r = k * F; for (let j = 0; j < F; j++) X[j] += a * PCA.C[r + j]; }
  let mx = -1e9; for (let j = 0; j < F; j++) if (X[j] > mx) mx = X[j];
  for (let j = 0; j < F; j++) X[j] = Math.max(X[j] - mx, -80);
  return X;
}
// the same simple measures as pitcher/xypad/latent.py
function measures(X) {
  const Tn = PCA.T, e = new Float32Array(Tn), P = new Float32Array(X.length);
  for (let j = 0; j < X.length; j++) P[j] = 10 ** (X[j] / 10);
  let emax = -1e9;
  for (let t = 0; t < Tn; t++) { let s = 0; for (let b = 0; b < NM; b++) s += P[t * NM + b]; e[t] = 10 * Math.log10(s + 1e-12); if (e[t] > emax) emax = e[t]; }
  const Pb = new Float64Array(NM); let na = 0;
  for (let t = 0; t < Tn; t++) if (e[t] >= emax - 30) { na++; for (let b = 0; b < NM; b++) Pb[b] += P[t * NM + b]; }
  let sp = 0, sf = 0, sl = 0;
  for (let b = 0; b < NM; b++) { Pb[b] /= Math.max(na, 1); sp += Pb[b]; sf += Pb[b] * LT.band_hz[b]; sl += Math.log(Pb[b] + 1e-12); }
  let att = 0; while (att < Tn && e[att] < emax - 3) att++;
  let last = Tn - 1; while (last > 0 && e[last] < emax - 40) last--;
  let dsum = 0, nl = 0;
  for (let t = 1; t < LT.hold_frames; t++) for (let b = 0; b < NM; b++) {
    const a = X[t * NM + b], p = X[(t - 1) * NM + b]; if (Math.max(a, p) > -50) { dsum += Math.abs(a - p); nl++; } }
  return { brightness: Math.log2(sf / sp / LT.f0), attack: att * LT.dt, length: (last + 1) * LT.dt,
           noise: Math.exp(sl / NM) / (sp / NM), movement: nl ? dsum / nl : 0 };
}

// ---------- readable parameter names ----------
const KNOB_GROUP = {}; D.groups.forEach(g => g.knobs.forEach(k => { KNOB_GROUP[k] = g.name; }));
const ENV = { a: "attack", d: "decay", s: "sustain", r: "release" };
function pname(id) {
  if (KNOB_GROUP[id]) return KNOB_GROUP[id];
  let m;
  if ((m = id.match(/^scene\/adsr\/([01])\/([adsr])$/))) return `${m[1] === "0" ? "Amp" : "Filter"} env ${ENV[m[2]]}`;
  if ((m = id.match(/^scene\/adsr\/([01])\/(\w)_s$/))) return `${m[1] === "0" ? "Amp" : "Filter"} env ${ENV[m[2]] || m[2]} shape`;
  if ((m = id.match(/^scene\/filterunit\/(\d)\/(\w+)$/))) return `Filter ${+m[1] + 1} ${m[2] === "envmod" ? "env amount" : m[2] === "model" ? "type" : m[2]}`;
  if ((m = id.match(/^scene\/osc\/(\d)\/type$/))) return `Osc ${+m[1] + 1} type`;
  if ((m = id.match(/^scene\/osc\/(\d)\/(\w+)\/p(\d+)/))) return `Osc ${+m[1] + 1} ${m[2]} param ${+m[3] + 1}`;
  if ((m = id.match(/^scene\/osc\/(\d)\/(\w+)/))) return `Osc ${+m[1] + 1} ${m[2]}`;
  if ((m = id.match(/^scene\/lfo\/(\d+)\/(\w+)/))) return `LFO ${+m[1] + 1} ${m[2]}`;
  if ((m = id.match(/^scene\/level_(\w+)$/))) return `${m[1].replace("o", "Osc ").replace("noise", "Noise")} level`;
  if ((m = id.match(/^scene\/mute_(\w+)$/))) return `${m[1].replace("o", "Osc ").replace("noise", "Noise")} mute`;
  if ((m = id.match(/^fx\/(\w+)\/present$/))) return `${m[1]} on`;
  if ((m = id.match(/^fx\/(\w+)\/(placement|position)$/))) return `${m[1]} ${m[2]}`;
  if ((m = id.match(/^fx\/(\w+)\/p(\d+)/))) return `${m[1]} param ${+m[2] + 1}`;
  if ((m = id.match(/^route\/([\w]+)\/(.+)$/))) return `mod ${m[1].replace(/_/g, " ")} → ${m[2].replace("scene/", "")}`;
  return id.replace("scene/", "");
}

// ---------- latent axes ----------
const RANGE = 5;                                   // the pad spans +-2.5 standard deviations
const fmtEff = {
  brightness: (v) => `${v >= 0 ? "brighter" : "darker"} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} oct`,
  attack: (v) => `${v >= 0 ? "slower" : "faster"} attack ${v >= 0 ? "+" : "−"}${Math.round(Math.abs(v) * 1000)} ms`,
  length: (v) => `${v >= 0 ? "longer" : "shorter"} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} s`,
  noise: (v) => `${v >= 0 ? "noisier" : "more tonal"}`,
  movement: (v) => `${v >= 0 ? "more movement" : "more static"}`,
};
const EFF_MIN = { brightness: 0.15, attack: 0.04, length: 0.15, noise: 0.02, movement: 0.04 };
const AX = [];
LT.directions.forEach(d => AX.push({ id: "s:" + d.key, kind: "sound", name: d.name, low: d.low, high: d.high, u: Float32Array.from(d.u), mean: d.mean, sd: d.sd,
  desc: `Follows the measured ${d.key} of the training sounds (R² ${d.r2.toFixed(2)} from z). Two standard deviations along it: ${Math.abs(d.effect) >= EFF_MIN[d.key] ? fmtEff[d.key](d.effect) : "a small change"} in the decoded sound.`, key: d.key }));
const MAIN_DB = 2;
LT.dims.forEach((r, rank) => {
  const u = new Float32Array(ZD); u[r.d] = 1;
  const secs = Object.entries(r.sections).sort((a, b) => b[1] - a[1]);
  const effs = Object.entries(r.effect).filter(([k, v]) => Math.abs(v) >= EFF_MIN[k]).sort((a, b) => Math.abs(b[1]) / EFF_MIN[b[0]] - Math.abs(a[1]) / EFF_MIN[a[0]]);
  const name = secs[0][1] > 0.17 ? `${secs[0][0]}${secs[1][1] > 0.15 ? " + " + secs[1][0].toLowerCase() : ""}` : "Mixed";
  const up = effs.length ? effs.slice(0, 3).map(([k, v]) => fmtEff[k](v)).join(", ") : "small change in the sound";
  AX.push({ id: "z:" + r.d, kind: "dim", d: r.d, rank, name: `z${r.d} · ${name}`, low: "−", high: "+", u, mean: r.mean, sd: r.sd, impact: r.impact, secs, effs,
    minor: r.impact < MAIN_DB,
    desc: `Up: ${up}. Most specific changes: ${r.top.slice(0, 3).map(pname).join(", ")}.` });
});
const AXI = Object.fromEntries(AX.map(a => [a.id, a]));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const tOf = (ax, pos) => ax.mean + (pos - 0.5) * RANGE * ax.sd;
const posOf = (ax, z) => (dot(ax.u, z) - ax.mean) / (RANGE * ax.sd) + 0.5;
function constrain(z0, axes, ts) {                  // smallest change of z so that axis i reads ts[i]
  const n = axes.length, z = Float32Array.from(z0); if (!n) return z;
  const G = axes.map((a, i) => axes.map((b, j) => dot(a.u, b.u) + (i === j ? 1e-6 : 0)));
  const r = axes.map((a, i) => ts[i] - dot(a.u, z0));
  for (let c = 0; c < n; c++) {                    // Gaussian elimination
    let p = c; for (let k = c + 1; k < n; k++) if (Math.abs(G[k][c]) > Math.abs(G[p][c])) p = k;
    [G[c], G[p]] = [G[p], G[c]]; [r[c], r[p]] = [r[p], r[c]];
    for (let k = c + 1; k < n; k++) { const f = G[k][c] / G[c][c]; for (let j = c; j < n; j++) G[k][j] -= f * G[c][j]; r[k] -= f * r[c]; }
  }
  const a = new Array(n).fill(0);
  for (let c = n - 1; c >= 0; c--) { let s = r[c]; for (let j = c + 1; j < n; j++) s -= G[c][j] * a[j]; a[c] = s / G[c][c]; }
  axes.forEach((ax, i) => { for (let k = 0; k < ZD; k++) z[k] += a[i] * ax.u[k]; });
  return z;
}
const PZ = f16(LT.presets.z), NPZ = LT.presets.label.length;
const pz = (i) => PZ.subarray(i * ZD, (i + 1) * ZD);
const LBASES = [...Object.entries(LT.mean_z).map(([l, z]) => ({ name: `Typical ${l}`, z: Float32Array.from(z), y: LT.labels.indexOf(l) })),
                { name: "Random real preset", random: true }];

// ---------- state ----------
const ranked = D.groups;
const byName = Object.fromEntries(ranked.map(g => [g.name, g]));
const state = { mode: "latent", base: 0, vals: BASES[0].vals.slice(), axes: { x: ranked[0].name, y: ranked[1].name, z: ranked[2].name }, pos: { x: 0.5, y: 0.5, z: 0.5 } };
const s0 = store.get("precision-xypad-axes"); if (s0 && byName[s0.x] && byName[s0.y]) state.axes = { x: s0.x, y: s0.y, z: byName[s0.z] ? s0.z : null };
const firstDims = AX.filter(a => a.kind === "dim");
const lat = { base: 0, z: Float32Array.from(LBASES[0].z), y: LBASES[0].y, axes: { x: "s:brightness", y: "s:length", z: firstDims[0].id }, pos: { x: 0.5, y: 0.5, z: 0.5 },
              lock: null, out: null, X: null, m: null, msg: "" };
const s1 = store.get("precision-xypad-latent-axes"); if (s1 && AXI[s1.x] && AXI[s1.y]) lat.axes = { x: s1.x, y: s1.y, z: AXI[s1.z] ? s1.z : null };
const sm = location.hash === "#knob" ? "knob" : location.hash === "#latent" ? "latent" : store.get("precision-xypad-mode"); if (sm === "knob" || sm === "latent") state.mode = sm;
const latAxes = () => ["x", "y", "z"].map(a => AXI[lat.axes[a]]).filter(Boolean);
function latSyncPos() { for (const a of ["x", "y", "z"]) { const ax = AXI[lat.axes[a]]; if (ax) lat.pos[a] = posOf(ax, lat.z); } }
function latApplyPos() {                           // pad position -> z (other directions unchanged)
  const keys = ["x", "y", "z"].filter(a => AXI[lat.axes[a]]);
  lat.z = constrain(lat.z, keys.map(a => AXI[lat.axes[a]]), keys.map(a => tOf(AXI[lat.axes[a]], lat.pos[a])));
}
function latDecode() {                             // z -> preset values for the synth + expected sound
  const o = decodeP(lat.z, lat.y);
  if (lat.lock) { o.act = lat.lock.act; o.bools = lat.lock.bools; o.cats = lat.lock.cats; }
  lat.out = o;
  for (let k = 0; k < NK; k++) state.vals[k] = o.act[k] ? o.num[k] : NaN;
  lat.X = decodeA(lat.z, lat.y); lat.m = measures(lat.X);
}
function syncPosFromVals() { for (const a of ["x", "y", "z"]) { const g = byName[state.axes[a]]; if (!g) continue; const p = groupPos(g, state.vals); state.pos[a] = p == null ? 0.5 : p; } }
function applyPos() { for (const a of ["x", "y", "z"]) { const g = byName[state.axes[a]]; if (g) setGroup(g, state.pos[a], state.vals); } }
const isLat = () => state.mode === "latent";
const P_ = () => (isLat() ? lat.pos : state.pos);

// ---------- toolbar ----------
const baseSel = document.getElementById("base"), catSel = document.getElementById("cat"), bgSel = document.getElementById("bg");
function fillBase() {
  baseSel.innerHTML = "";
  (isLat() ? LBASES : BASES).forEach((b, i) => { const o = document.createElement("option"); o.value = i; o.textContent = b.name; baseSel.appendChild(o); });
  baseSel.value = isLat() ? lat.base : state.base;
  bgSel.innerHTML = "";
  const opts = isLat() ? [["cat", "Category guess"], ["brightness", "Brightness"], ["length", "Length"], ["attack", "Attack"], ["noise", "Noise"], ["none", "None"]] : [["cat", "Category guess"], ["none", "None"]];
  opts.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; bgSel.appendChild(o); });
  const want = store.get(isLat() ? "precision-xypad-bg-latent" : "precision-xypad-bg-knob");
  bgSel.value = opts.some(o => o[0] === want) ? want : (isLat() ? "brightness" : "cat");
}
LT.labels.forEach((l, i) => { const o = document.createElement("option"); o.value = i; o.textContent = l; catSel.appendChild(o); });
function loadLatBase(i) {
  const b = LBASES[i]; lat.base = i;
  if (b.random) { const k = Math.floor(Math.random() * NPZ); lat.z = Float32Array.from(pz(k)); lat.y = LT.presets.label[k]; lat.msg = `Started from “${LT.presets.name[k] || "preset " + k}” (${LT.labels[lat.y]}).`; }
  else { lat.z = Float32Array.from(b.z); lat.y = b.y; lat.msg = ""; }
  catSel.value = lat.y; lat.lock = null; lockBtn.setAttribute("aria-pressed", "false");
  latSyncPos(); latDecode();
}
baseSel.addEventListener("change", () => {
  if (isLat()) loadLatBase(+baseSel.value);
  else { state.base = +baseSel.value; state.vals = BASES[state.base].vals.slice(); syncPosFromVals(); }
  mapDirty = true; render();
});
catSel.addEventListener("change", () => { lat.y = +catSel.value; latDecode(); mapDirty = true; render(); });
document.getElementById("reset").addEventListener("click", () => {
  if (isLat()) loadLatBase(lat.base); else { state.vals = BASES[state.base].vals.slice(); syncPosFromVals(); }
  mapDirty = true; render();
});
const lockBtn = document.getElementById("lock");
lockBtn.addEventListener("click", () => {
  if (lat.lock) lat.lock = null; else { const o = decodeP(lat.z, lat.y); lat.lock = { act: o.act, bools: o.bools, cats: o.cats }; }
  lockBtn.setAttribute("aria-pressed", String(!!lat.lock)); latDecode(); mapDirty = true; render(false);
});
bgSel.addEventListener("change", () => { store.set(isLat() ? "precision-xypad-bg-latent" : "precision-xypad-bg-knob", bgSel.value); mapDirty = true; render(false); });
function setMode(m) {
  state.mode = m; store.set("precision-xypad-mode", m);
  document.getElementById("m-latent").setAttribute("aria-pressed", String(m === "latent"));
  document.getElementById("m-knob").setAttribute("aria-pressed", String(m === "knob"));
  document.querySelectorAll(".only-latent").forEach(e => { e.hidden = m !== "latent"; });
  document.querySelectorAll(".only-knob").forEach(e => { e.hidden = m !== "knob"; });
  if (m === "latent") latDecode(); else syncPosFromVals();   // knob pad starts from the decoded preset
  fillBase(); mapDirty = true; render();
}
document.getElementById("m-latent").addEventListener("click", () => setMode("latent"));
document.getElementById("m-knob").addEventListener("click", () => setMode("knob"));
const legend = document.getElementById("legend");

// ---------- list ----------
const list = document.getElementById("list");
const AUDIBLE = new Set(["Amp attack", "Amp decay", "Amp sustain", "Amp release", "Filter env attack", "Filter env decay", "Filter env sustain", "Filter env release",
  "Cutoff", "Resonance", "Filter env amount", "Filter keytrack", "Drive", "Glide", "Drift", "Osc 2/3 detune", "Osc 1 level", "Osc 2/3 level", "Noise level",
  "Classic shape", "Sub mix", "Unison detune", "LFO 1 rate", "LFO 1 amount", "Reverb size / decay", "Reverb mix", "Delay time", "Delay feedback", "Delay mix", "Volume"]);
const maxImp = Math.max(...ranked.map(g => g.importance));
const maxLat = Math.max(...AX.filter(a => a.kind === "dim").map(a => a.impact));
const axButtons = (on, label) => ["x", "y", "z"].map(a => `<button class="axbtn" type="button" data-a="${a}" aria-pressed="${on.includes(a)}" aria-label="Put ${label} on ${a.toUpperCase()}">${a.toUpperCase()}</button>`).join("");
function drawKnobList(q) {
  document.getElementById("listtitle").textContent = "Controls";
  document.getElementById("listhint").textContent = "Ranked by how much the category guess drops when the group is scrambled (balanced accuracy, 8 categories). A grey bar means the effect is within the noise.";
  ranked.forEach((g, r) => {
    if (q && !(g.name + " " + g.section + " " + g.desc + " " + g.knobs.join(" ")).toLowerCase().includes(q)) return;
    const on = Object.entries(state.axes).filter(([, n]) => n === g.name).map(([a]) => a);
    const clear = g.importance > 2 * g.se, w = Math.max(0, g.importance) / maxImp * 100;
    const el = document.createElement("div");
    el.className = "g" + (on.length ? " assigned" : "");
    el.innerHTML = `<div class="rank">${r + 1}</div>
      <div class="nm">${g.name}<span class="sec">${g.section}</span></div>
      <div class="axbtns">${axButtons(on, g.name)}</div>
      <div class="meter"><div class="track"><i class="${clear ? "" : "unclear"}" style="width:${w.toFixed(1)}%"></i></div>
        <span class="num">${g.importance >= 0 ? "+" : ""}${(g.importance * 100).toFixed(1)} pts</span></div>
      <div class="tags"><span class="tag">${g.knobs.length} knob${g.knobs.length > 1 ? "s" : ""}</span>
        ${g.mean_abs_rho != null ? `<span class="tag">move together: |ρ| ${g.mean_abs_rho.toFixed(2)}</span>` : ""}
        <span class="tag">in ${Math.round(g.active * 100)}% of presets</span>
        ${AUDIBLE.has(g.name) ? "" : `<span class="tag">no preview sound</span>`}</div>`;
    el.querySelectorAll(".axbtn").forEach(b => b.addEventListener("click", () => assign(b.dataset.a, g.name)));
    list.appendChild(el);
  });
}
function latItem(ax, rank) {
  const on = Object.entries(lat.axes).filter(([, n]) => n === ax.id).map(([a]) => a);
  const el = document.createElement("div");
  el.className = "g" + (on.length ? " assigned" : "");
  const p = Math.min(1, Math.max(0, posOf(ax, lat.z)));
  let meter = "", tags = "";
  if (ax.kind === "dim") {
    meter = `<div class="meter"><div class="track"><i class="${ax.minor ? "unclear" : ""}" style="width:${(ax.impact / maxLat * 100).toFixed(1)}%"></i></div><span class="num">${ax.impact.toFixed(1)} dB</span></div>`;
    tags = ax.secs.slice(0, 3).map(([s, v], i) => `<span class="tag${i === 0 && v > 0.17 ? " strong" : ""}">${s} ${Math.round(v * 100)}%</span>`).join("");
  } else tags = `<span class="tag">sound axis</span>`;
  el.innerHTML = `<div class="rank">${rank}</div>
    <div class="nm">${ax.name}</div>
    <div class="axbtns">${axButtons(on, ax.name)}</div>
    ${meter}
    <div class="tags">${tags}</div>
    <div class="slide"><span>${ax.low}</span><input type="range" min="0" max="1000" value="${Math.round(p * 1000)}" aria-label="${ax.name}"><span>${ax.high}</span></div>`;
  el.querySelectorAll(".axbtn").forEach(b => b.addEventListener("click", () => assign(b.dataset.a, ax.id)));
  const sl = el.querySelector("input");
  sl.addEventListener("input", () => {           // move along this axis, keep the pad's axes where they are
    const assigned = ["x", "y", "z"].filter(a => AXI[lat.axes[a]] && lat.axes[a] !== ax.id);
    const axes = [ax, ...assigned.map(a => AXI[lat.axes[a]])], ts = [tOf(ax, sl.value / 1000), ...assigned.map(a => tOf(AXI[lat.axes[a]], lat.pos[a]))];
    lat.z = constrain(lat.z, axes, ts); latSyncPos(); latDecode(); render(false); refreshSliders(sl);
    clearTimeout(mapTimer); mapTimer = setTimeout(() => { mapDirty = true; render(false); }, 150);
  });
  el._ax = ax;
  return el;
}
function refreshSliders(except) {
  list.querySelectorAll(".g").forEach(el => { if (!el._ax) return; const s = el.querySelector("input"); if (s !== except) s.value = Math.round(Math.min(1, Math.max(0, posOf(el._ax, lat.z))) * 1000); });
}
function drawLatList() {
  AX.filter(a => a.kind === "sound").forEach((a, i) => list.appendChild(latItem(a, i + 1)));
  AX.filter(a => a.kind === "dim").forEach(a => list.appendChild(latItem(a, a.rank + 1)));
}
function drawList() {
  list.innerHTML = "";
  if (isLat()) drawLatList(); else drawKnobList("");
}
function assign(a, name) {
  const ax = isLat() ? lat.axes : state.axes;
  if (ax[a] === name) { if (a === "z") ax.z = null; else return; }
  else { for (const o of ["x", "y", "z"]) if (o !== a && ax[o] === name) ax[o] = ax[a]; ax[a] = name; }
  if (!ax.x || !ax.y) return;
  if (isLat()) { store.set("precision-xypad-latent-axes", lat.axes); latSyncPos(); }
  else { store.set("precision-xypad-axes", state.axes); syncPosFromVals(); }
  mapDirty = true; render();
}

// ---------- pad ----------
const cv = document.getElementById("pad"), ctx = cv.getContext("2d");
const GRID = 40, LGRID = 24;
let padView = 'xy';
let highlightedCategory = null;
const orbitCamera = XyzPad.createCamera();
let cameraStart = null, orbitGesture = null, handleStart = null, handleSelected = false;
const hasZ = () => isLat() ? Boolean(AXI[lat.axes.z]) : Boolean(byName[state.axes.z]);
function xyzDots() {
  if (isLat()) {
    const axes = ['x','y','z'].map(a => AXI[lat.axes[a]]);
    if (axes.some(a => !a)) return [];
    return Array.from({length: NPZ}, (_,i) => [...axes.map(a => posOf(a,pz(i))), LT.presets.label[i]]);
  }
  const groups = ['x','y','z'].map(a => byName[state.axes[a]]);
  if (groups.some(g => !g)) return [];
  return PV.map((row,i) => [...groups.map(g => groupPos(g,row)), PL[i]])
    .filter(row => row.slice(0,3).every(v => v !== null));
}
function setPadView(view) {
  padView = view;
  bgSel.disabled = view !== 'xy';
  bgSel.title = view !== 'xy' ? 'Heat maps are available in the XY view' : '';
  if(view !== 'xy' && !hasZ()) {
    const active = isLat() ? lat.axes : state.axes;
    const choice = isLat() ? AX.find(a => a.id !== active.x && a.id !== active.y)?.id : ranked.find(g => g.name !== active.x && g.name !== active.y)?.name;
    if(choice) assign('z',choice);
  }
  document.getElementById('camera-controls').hidden = view!=='orbit';
  document.getElementById('xyz-help').hidden = view==='xy';
  document.querySelector('.padwrap').classList.toggle('xyz-view',view!=='xy');
  cv.setAttribute('aria-label', view==='orbit' ? 'Preset cloud. Drag the handle to move sound; Shift-drag it for depth. Click it for axis arrows. Drag empty space or use arrow keys to rotate the camera. Scroll to zoom. Home resets the camera.' : view==='xyz' ? 'XYZ pad. Drag for X and Y. Shift-drag or Page Up and Page Down for Z. The Z slider also controls depth.' : 'XY pad: drag, or use the arrow keys');
  const url=new URL(location.href);url.searchParams.set('view',view);history.replaceState(null,'',url);
  render(false);
}
function zoomCamera(factor) {orbitCamera.zoom=Math.max(.5,Math.min(2.5,orbitCamera.zoom*factor));drawPad();}
document.getElementById('camera-in').addEventListener('click',()=>zoomCamera(1.15));
document.getElementById('camera-out').addEventListener('click',()=>zoomCamera(1/1.15));
document.querySelectorAll('[data-camera]').forEach(button=>button.addEventListener('click',()=>{
  const views={front:{yaw:0,pitch:0},top:{yaw:0,pitch:Math.PI/2},side:{yaw:-Math.PI/2,pitch:0},reset:XyzPad.createCamera()};
  Object.assign(orbitCamera,views[button.dataset.camera]);drawPad();
}));
cv.addEventListener('wheel',e=>{if(padView!=='orbit')return;e.preventDefault();zoomCamera(Math.exp(-Math.max(-100,Math.min(100,e.deltaY))*.002));},{passive:false});
let mapDirty = true, mapImg = null, dotsCache = null, mapTimer = null;
function resize() { const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); render(false); }
function computeKnobMap() {
  const gx = byName[state.axes.x], gy = byName[state.axes.y];
  const vals = state.vals.slice(), cells = new Uint8Array(GRID * GRID), conf = new Float32Array(GRID * GRID);
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    setGroup(gx, (i + 0.5) / GRID, vals); setGroup(gy, 1 - (j + 0.5) / GRID, vals);
    const p = predict(vals), b = argmax(p); cells[j * GRID + i] = b; conf[j * GRID + i] = p[b];
  }
  mapImg = { n: GRID, kind: bgSel.value === "none" ? "none" : "cat", cells, conf };
  const xs = PV.map(r => groupPos(gx, r)), ys = PV.map(r => groupPos(gy, r));
  dotsCache = PV.map((_, i) => (xs[i] == null || ys[i] == null) ? null : [xs[i], ys[i], PL[i], -1]).filter(Boolean);
}
function computeLatMap() {
  const ax = AXI[lat.axes.x], ay = AXI[lat.axes.y], az = AXI[lat.axes.z], kind = bgSel.value;
  const n = LGRID, cells = new Uint8Array(n * n), conf = new Float32Array(n * n), val = new Float32Array(n * n);
  const axes = [ax, ay].concat(az ? [az] : []);
  if (kind !== "none") {
    const vals = new Array(NK);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const ts = [tOf(ax, (i + 0.5) / n), tOf(ay, 1 - (j + 0.5) / n)].concat(az ? [tOf(az, lat.pos.z)] : []);
      const z = constrain(lat.z, axes, ts), k = j * n + i;
      if (kind === "cat") {
        const o = decodeP(z, lat.y), act = lat.lock ? lat.lock.act : o.act;
        for (let q = 0; q < NK; q++) vals[q] = act[q] ? o.num[q] : NaN;
        const p = predict(vals), b = argmax(p); cells[k] = b; conf[k] = p[b];
      } else val[k] = measures(decodeA(z, lat.y))[kind];
    }
  }
  let lo = Infinity, hi = -Infinity;
  if (kind !== "cat" && kind !== "none") { const s = Array.from(val).sort((a, b) => a - b); lo = s[Math.floor(s.length * 0.02)]; hi = s[Math.floor(s.length * 0.98)]; }
  mapImg = { n, kind, cells, conf, val, lo, hi };
  dotsCache = [];
  for (let i = 0; i < NPZ; i++) { const z = pz(i); dotsCache.push([posOf(ax, z), posOf(ay, z), LT.presets.label[i], i]); }
}
function computeMap() { if (isLat()) computeLatMap(); else computeKnobMap(); mapDirty = false; }
function hexA(hex, a) { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
const MEAS_WORDS = { brightness: ["dark", "bright"], length: ["short", "long"], attack: ["snappy", "slow attack"], noise: ["tonal", "noisy"] };
function drawLegend() {
  const k = mapImg ? mapImg.kind : "cat";
  if (k === "cat") legend.innerHTML = L.map((l, i) => `<span><i class="sw" style="background:${COL[i]}"></i>${l}</span>`).join("");
  else if (k === "none") legend.innerHTML = "";
  else { const w = MEAS_WORDS[k]; legend.innerHTML = `<span>${w[0]}</span><i class="ramp" style="background:linear-gradient(90deg, ${hexA(cssVar("--heat"), 0.03)}, ${hexA(cssVar("--heat"), 0.75)})"></i><span>${w[1]}</span>`; }
  if (k !== "none" && (document.getElementById("showdots").checked)) legend.innerHTML += `<span class="mono">Preset categories</span>`;
}
function drawPad() {
  if(padView!=='xy') {
    if(!hasZ()) { document.getElementById('xyz-help').textContent='Assign a Z axis to explore depth'; }
    else document.getElementById('xyz-help').textContent=padView==='orbit'?'Handle: move · Shift: depth · Empty space: orbit':'Drag XY · Shift-drag depth';
    XyzPad.draw(ctx,cv.width,cv.height,{point:{...P_(),z:hasZ()?P_().z:0},dots:xyzDots(),accent:cssVar('--accent'),ink:cssVar('--muted'),line:cssVar('--line'),colors:COL,category:highlightedCategory ?? (isLat()?lat.y:null),highlightCategory:highlightedCategory!==null,showDots:document.getElementById('showdots').checked,camera:padView==='orbit'?orbitCamera:null,selected:handleSelected});
    legend.textContent=padView==='orbit'?'Click handle for XYZ arrows · scroll to zoom':'Preset positions in XYZ · highlighted plane is the current depth';
    return;
  }
  if (mapDirty) computeMap();
  const W = cv.width, H = cv.height, dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, W, H);
  const alpha = parseFloat(cssVar("--map-alpha")) || 0.3, M = mapImg, n = M.n;
  const cw = W / n, ch = H / n;
  const paint = (rgba) => {                        // one pixel per cell, scaled up smoothly (no seams between cells)
    const off = document.createElement("canvas"); off.width = n; off.height = n; const g = off.getContext("2d"), img = g.createImageData(n, n);
    for (let k = 0; k < n * n; k++) img.data.set(rgba(k), k * 4);
    g.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
  };
  const rgb = (hex) => { const h = hex.replace("#", ""), v = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; };
  if (M.kind === "cat") {
    const cols = COL.map(rgb);
    paint((k) => [...cols[M.cells[k]], Math.round(255 * Math.min(1, alpha * (0.45 + 0.9 * M.conf[k])))]);
    const cnt = new Array(NL).fill(0), sx = new Array(NL).fill(0), sy = new Array(NL).fill(0);
    M.cells.forEach((c, k) => { cnt[c]++; sx[c] += k % n; sy[c] += Math.floor(k / n); });
    ctx.font = `600 ${12 * dpr}px ${cssVar("--display").split(",")[0]}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let c = 0; c < NL; c++) if (cnt[c] > n * n * 0.04) {
      ctx.fillStyle = cssVar("--ink"); ctx.globalAlpha = 0.75; ctx.fillText(L[c], (sx[c] / cnt[c] + 0.5) * cw, (sy[c] / cnt[c] + 0.5) * ch); ctx.globalAlpha = 1; }
  } else if (M.kind !== "none" && M.kind !== "cat") {
    const heat = rgb(cssVar("--heat"));
    paint((k) => { const t = Math.min(1, Math.max(0, (M.val[k] - M.lo) / ((M.hi - M.lo) || 1))); return [...heat, Math.round(255 * (0.02 + 0.30 * t))]; });
  }
  ctx.strokeStyle = cssVar("--line"); ctx.lineWidth = 1;
  for (let q = 1; q < 4; q++) { ctx.beginPath(); ctx.moveTo(W * q / 4, 0); ctx.lineTo(W * q / 4, H); ctx.moveTo(0, H * q / 4); ctx.lineTo(W, H * q / 4); ctx.stroke(); }
  if (document.getElementById("showdots").checked && dotsCache) {
    const ring = cssVar("--dot-ring");
    for (const [x, y, l] of dotsCache) {
      if (x < -0.02 || x > 1.02 || y < -0.02 || y > 1.02) continue;
      const same = !isLat() || l === lat.y;
      ctx.fillStyle = hexA(COL[l], same ? 0.85 : 0.28); ctx.beginPath(); ctx.arc(x * W, (1 - y) * H, (same ? 2.6 : 1.8) * dpr, 0, 6.283); ctx.fill();
      if (same && isLat()) { ctx.strokeStyle = hexA(ring, 0.35); ctx.lineWidth = 0.8 * dpr; ctx.stroke(); }
    }
  }
  const p = P_(), cx = Math.min(1, Math.max(0, p.x)) * W, cy = (1 - Math.min(1, Math.max(0, p.y))) * H;
  ctx.strokeStyle = cssVar("--dot-ring"); ctx.lineWidth = 2 * dpr; ctx.fillStyle = cssVar("--well");
  ctx.beginPath(); ctx.arc(cx, cy, 9 * dpr, 0, 6.283); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 2.5 * dpr, 0, 6.283); ctx.fillStyle = cssVar("--dot-ring"); ctx.fill();
  drawLegend();
}
let dragging = false, downAt = null, depthStart = 0;
function padSet(e) {
  if(padView==='orbit') {
    if(downAt && cameraStart){
      if(orbitGesture==='camera'){
        orbitCamera.yaw=cameraStart.yaw+(e.clientX-downAt[0])*.008;orbitCamera.pitch=cameraStart.pitch+(e.clientY-downAt[1])*.008;drawPad();
      } else if(handleStart && orbitGesture){
        const r=cv.getBoundingClientRect();
        const next=XyzPad.orbitDrag(handleStart,e.clientX-downAt[0],e.clientY-downAt[1],cameraStart,r.width,r.height,orbitGesture);
        if(!hasZ())next.z=handleStart.z;
        const current=P_();
        if(['x','y','z'].some(a=>Math.abs(next[a]-current[a])>1e-8)){
          Object.assign(current,next);mapDirty=true;
          if(isLat()){latApplyPos();latDecode();refreshSliders();}else applyPos();
          render(false);
        }else drawPad();
      }
    }
    return;
  }
  const r = cv.getBoundingClientRect(), p = P_();
  if(padView==='xyz') {
    if(e.shiftKey && downAt && hasZ()) p.z=Math.min(1,Math.max(0,depthStart+(downAt[1]-e.clientY)/r.height));
    else Object.assign(p,XyzPad.unproject((e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height,hasZ()?p.z:0));
    mapDirty=true;
  } else {p.x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); p.y = Math.min(1, Math.max(0, 1 - (e.clientY - r.top) / r.height));}
  if (isLat()) { latApplyPos(); latDecode(); } else applyPos();
  render(false); if (isLat()) refreshSliders();
}
function nearestDot(e) {
  if (!dotsCache || !document.getElementById("showdots").checked) return null;
  const r = cv.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = 1 - (e.clientY - r.top) / r.height;
  let best = null, bd = (7 / r.width) ** 2;
  for (const d of dotsCache) { if (isLat() && d[2] !== lat.y) continue; const dd = (d[0] - x) ** 2 + (d[1] - y) ** 2; if (dd < bd) { bd = dd; best = d; } }
  return best;
}
cv.addEventListener("pointerdown", e => {
  if(e.button!==0)return;
  dragging=true;downAt=[e.clientX,e.clientY];depthStart=P_().z;cameraStart={...orbitCamera};
  if(padView==='orbit'){
    const r=cv.getBoundingClientRect(),p=P_();
    handleStart={x:Math.max(0,Math.min(1,p.x)),y:Math.max(0,Math.min(1,p.y)),z:hasZ()?Math.max(0,Math.min(1,p.z)):0};
    const hit=XyzPad.hitOrbit(handleStart,orbitCamera,r.width,r.height,e.clientX-r.left,e.clientY-r.top,handleSelected);
    orbitGesture=hit==='handle'?(e.shiftKey?'depth':'plane'):hit||'camera';
    handleSelected=Boolean(hit);
  }
  cv.focus();cv.setPointerCapture(e.pointerId);
  if(padView==='orbit')drawPad();else padSet(e);
});
cv.addEventListener("pointermove", e => { if (dragging) padSet(e); });
cv.addEventListener("pointerup", e => {
  dragging = false;
  if (padView==='xy' && isLat() && downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 3) {
    const d = nearestDot(e);
    if (d) { lat.z = Float32Array.from(pz(d[3])); lat.y = d[2]; catSel.value = lat.y; lat.msg = `Jumped to the real preset “${LT.presets.name[d[3]] || "preset " + d[3]}” (${LT.labels[d[2]]}).`;
      latSyncPos(); latDecode(); mapDirty = true; render(false); refreshSliders(); }
  }
  downAt = null; orbitGesture=null; handleStart=null;
});
cv.addEventListener("keydown", e => {
  if(padView==='orbit') {
    const step=e.shiftKey?.2:.07;
    if(e.key==='ArrowLeft')orbitCamera.yaw-=step;
    else if(e.key==='ArrowRight')orbitCamera.yaw+=step;
    else if(e.key==='ArrowUp')orbitCamera.pitch-=step;
    else if(e.key==='ArrowDown')orbitCamera.pitch+=step;
    else if(e.key==='Home')Object.assign(orbitCamera,XyzPad.createCamera());
    else if(e.key==='+'||e.key==='='){e.preventDefault();zoomCamera(1.15);return;}
    else if(e.key==='-'){e.preventDefault();zoomCamera(1/1.15);return;}
    else return;
    e.preventDefault();drawPad();return;
  }
  const d = e.shiftKey ? 0.1 : 0.02, m = { ArrowLeft: ["x", -d], ArrowRight: ["x", d], ArrowUp: ["y", d], ArrowDown: ["y", -d], PageUp: ["z", d], PageDown: ["z", -d] }[e.key];
  if (!m || (m[0] === "z" && !hasZ())) return; e.preventDefault(); if(m[0] === "z") mapDirty=true; const p = P_(); p[m[0]] = Math.min(1, Math.max(0, p[m[0]] + m[1]));
  if (isLat()) { latApplyPos(); latDecode(); refreshSliders(); } else applyPos(); render(false); });
cv.addEventListener("pointercancel", () => { dragging=false; downAt=null; orbitGesture=null; handleStart=null; });
const zEl = document.getElementById("z");
zEl.addEventListener("input", () => { const p = P_(); p.z = zEl.value / 1000;
  if (isLat()) { latApplyPos(); latDecode(); refreshSliders(); } else applyPos();
  clearTimeout(mapTimer); mapTimer = setTimeout(() => { mapDirty = true; render(false); }, 120); render(false); });
document.getElementById("showdots").addEventListener("change", () => render(false));

// ---------- readouts ----------
function fmt(name, vals) {
  const i = KI[name], k = D.knobs[i], v = vals[i]; if (v == null || Number.isNaN(v)) return "off";
  const x = nat(name, vals), t = k.ctrltype;
  if (t.startsWith("ct_envtime") || t === "ct_portatime") return (2 ** x < 1 ? `${Math.round(2 ** x * 1000)} ms` : `${(2 ** x).toFixed(2)} s`);
  if (t === "ct_reverbtime") return `${(2 ** x).toFixed(2)} s`;
  if (t.startsWith("ct_lforate")) return `${(2 ** x).toFixed(2)} Hz`;
  if (t.startsWith("ct_freq_audible")) return `${Math.round(440 * 2 ** (x / 12))} Hz`;
  if (t.startsWith("ct_pitch") || t === "ct_freq_mod") return `${x >= 0 ? "+" : ""}${x.toFixed(1)} st`;
  if (t.startsWith("ct_decibel")) return `${x >= 0 ? "+" : ""}${x.toFixed(1)} dB`;
  if (t.includes("bipolar")) return `${x >= 0 ? "+" : ""}${Math.round(x * 100)}%`;
  return `${Math.round(v * 100)}%`;
}
function drawAxes() {
  if (isLat()) {
    const ax = AXI[lat.axes.x], ay = AXI[lat.axes.y], az = AXI[lat.axes.z];
    document.getElementById("xaxis").innerHTML = `<span>${ax.low}</span><b>X · ${ax.name}</b><span>${ax.high}</span>`;
    document.getElementById("yaxis").innerHTML = `<span>${ay.low}</span><b>Y · ${ay.name}</b><span>${ay.high}</span>`;
    document.getElementById("zrow").hidden = !az;
    if (az) { document.getElementById("zname").textContent = az.name; zEl.value = Math.round(Math.min(1, Math.max(0, lat.pos.z)) * 1000); }
    return;
  }
  const ax = byName[state.axes.x], ay = byName[state.axes.y], az = byName[state.axes.z];
  document.getElementById("xaxis").innerHTML = `<span>low</span><b>X · ${ax.name}</b><span>high</span>`;
  document.getElementById("yaxis").innerHTML = `<span>low</span><b>Y · ${ay.name}</b><span>high</span>`;
  document.getElementById("zrow").hidden = !az;
  if (az) { document.getElementById("zname").textContent = az.name; zEl.value = Math.round(state.pos.z * 1000); }
}
const FX_NAMES = { "fx/reverb/present": "reverb", "fx/reverb2/present": "reverb 2", "fx/delay/present": "delay", "fx/chorus/present": "chorus", "fx/flanger/present": "flanger",
  "fx/phaser/present": "phaser", "fx/distortion/present": "distortion", "fx/eq/present": "EQ", "fx/rotary/present": "rotary", "fx/resonator/present": "resonator",
  "fx/freqshift/present": "freq shift", "fx/conditioner/present": "conditioner", "fx/nimbus/present": "nimbus" };
function specCanvas() {
  const c = document.getElementById("spec"); if (!c || !lat.X) return;
  const Tn = PCA.T; c.width=Tn; c.height=NM;
  const g = c.getContext("2d"), img = g.createImageData(Tn, NM);
  const hex = cssVar("--heat").replace("#", ""), n = parseInt(hex, 16), R = n >> 16 & 255, G = n >> 8 & 255, B = n & 255;
  const bgc = cssVar("--well").replace("#", ""), nb = parseInt(bgc.length === 3 ? bgc.split("").map(ch => ch + ch).join("") : bgc, 16), r0 = nb >> 16 & 255, g0 = nb >> 8 & 255, b0 = nb & 255;
  for (let t = 0; t < Tn; t++) for (let b = 0; b < NM; b++) {
    const v = Math.min(1, Math.max(0, (lat.X[t * NM + b] + 70) / 70)) ** 0.8, k = ((NM - 1 - b) * Tn + t) * 4;
    img.data[k] = r0 + (R - r0) * v; img.data[k + 1] = g0 + (G - g0) * v; img.data[k + 2] = b0 + (B - b0) * v; img.data[k + 3] = 255; }
  g.putImageData(img, 0, 0);
}
document.getElementById('probs').addEventListener('click', event => {
  const button=event.target.closest('[data-highlight-category]');
  if(!button)return;
  const category=Number(button.dataset.highlightCategory);
  highlightedCategory=highlightedCategory===category?null:category;
  document.querySelectorAll('[data-highlight-category]').forEach(item=>item.setAttribute('aria-pressed',String(Number(item.dataset.highlightCategory)===highlightedCategory)));
  drawPad(); // Visual focus only: preserve the sound and latent position.
});
function drawReadouts() {
  const p = predict(state.vals), top = argmax(p);
  document.getElementById("probs").innerHTML = `<div class="match-heading"><span>Category match</span><strong>Closest: ${L[top]} · ${Math.round(p[top] * 100)}%</strong></div>` + L.map((l, c) =>
    `<button type="button" data-highlight-category="${c}" aria-pressed="${highlightedCategory===c}" class="prob${c === top ? " top" : ""}"><span class="name">${l}</span><span class="bar"><i style="width:${(p[c] * 100).toFixed(1)}%;background:${COL[c]}"></i></span><span class="pct">${Math.round(p[c] * 100)}%</span></button>`).join("");
  const kn = document.getElementById("knobs");
  if (isLat() && lat.out) {
    const m = lat.m, o = lat.out;
    const osc = LT.cats.map((c, i) => `<div class="row wide"><span class="k">Osc ${i + 1}</span><span class="v">${c.options[o.cats[i]]}${o.bools[LT.bools.indexOf(`scene/mute_o${i + 1}`)] ? " (muted)" : ""}</span></div>`).join("");
    const fx = LT.bools.filter((b, j) => FX_NAMES[b] && o.bools[j]).map(b => FX_NAMES[b]);
    const row = (k, v) => `<div class="row wide"><span class="k">${k}</span><span class="v">${v}</span></div>`;
    kn.innerHTML = `<div class="eyebrow">Expected sound at the dot</div>
      ${row("Brightness", `${m.brightness.toFixed(1)} oct above the note`)}${row("Attack", m.attack < 0.05 ? "< 50 ms" : `${Math.round(m.attack * 1000)} ms`)}
      ${row("Length", `${m.length.toFixed(1)} s`)}${row("Noise", `${Math.round(m.noise * 100)}%`)}
      <div class="eyebrow" style="margin-top:8px">Decoded preset${lat.lock ? " · structure locked" : ""}</div>
      ${osc}${row("Effects", fx.length ? fx.join(", ") : "none")}
      ${row("Cutoff", fmt("scene/filterunit/0/cutoff", state.vals))}${row("Resonance", fmt("scene/filterunit/0/resonance", state.vals))}
      ${row("Amp A / D / S / R", ["a", "d", "s", "r"].map(x => fmt(`scene/adsr/0/${x}`, state.vals)).join(" · "))}`;
    specCanvas();
    return;
  }
  const rows = [];
  for (const a of ["x", "y", "z"]) { const g = byName[state.axes[a]]; if (!g) continue;
    for (const n of g.knobs) rows.push(`<div class="row"><span class="ax">${a.toUpperCase()}</span><span class="k">${n.replace("scene/", "")}</span><span class="v">${fmt(n, state.vals)}</span></div>`); }
  kn.innerHTML = `<div class="eyebrow">Knobs moved by the axes</div>` + rows.join("");
}
function drawHint() {
  document.getElementById("hint").textContent = isLat()
    ? (lat.msg ? lat.msg + " " : "") + "Drag the dot: every position is decoded into a preset. Click a dot to jump to that real preset. Turn the sound on and hold a key (A–K) while dragging."
    : "Turn the sound on and hold a key (or A–K on your computer keyboard) while dragging the dot. Glide is heard when you play a second key while holding the first.";
}
function render(full = true) {
  if (full) { drawAxes(); drawList(); }
  else if (isLat()) drawAxes();
  drawPad(); drawReadouts(); drawHint(); synthUpdate();
  window.dispatchEvent(new Event("xypad-patch-change"));
}

const r2s = LT.directions.map(d => `${d.key} ${d.r2.toFixed(2)}`).join(", ");
const nMain = AX.filter(a => a.kind === "dim" && !a.minor).length;
document.getElementById("method").innerHTML = [
  `<b>Latent pad:</b> the joint VAE (${LT.run}, z = ${ZD} numbers) trained on ${LT.n_rows.toLocaleString()} Surge presets with their sounds. The page runs its decoder directly: the dot's position gives z, z gives the preset.`,
  `<b>Sound axes:</b> for each training sound we measured brightness, attack, length, noise and movement, then found the direction in z that follows each one best (R² from z: ${r2s}).`,
  `<b>Latent dimensions:</b> each dimension was moved from its 10th to its 90th percentile for 400 real presets and decoded. The dB number is how much the expected sound changes; the tags show which synth sections it changes more than the other dimensions do. ${nMain} of ${LT.active.length} dimensions change the sound by more than ${MAIN_DB} dB.`,
  `<b>Honest limits:</b> raw dimensions mix several sections (no axis is purely "filter"); the expected sound comes from the VAE's own sound decoder, and the decoded parameters don't yet reproduce it reliably in Surge (decoder round trip 1–4% same character). The preview synth is a rough browser stand-in, not Surge.`,
  `<b>Knob pad:</b> ${D.n_presets.toLocaleString()} cleaned presets, ${NK} continuous knobs; groups ranked by a random forest (${Math.round(D.rf_acc * 100)}% balanced accuracy), category map from a smaller forest (${Math.round(D.page_acc * 100)}%).`,
].map(s => `<li>${s}</li>`).join("");
document.getElementById("cpp").innerHTML = [
  `<b>Latent pad = decoder in the plugin.</b> Pad position → z (move z along the chosen axes) → VAE decoder (ONNX, ~5 MB, &lt; 1 ms) → parameters. Fast enough to follow the cursor live.`,
  `<b>Axes are a table:</b> each axis is a direction in z (32 numbers) plus a mean, a spread and labels. Sound axes and latent dimensions use the same format, so better axes later (word directions from the prompt data, or a VAE with section blocks) only change the table.`,
  `<b>Lock structure:</b> keeps switches and choices (oscillator types, effects on/off) from the moment of locking; only continuous knobs follow the dot, so dragging never jumps between oscillator types.`,
  `<b>Activity:</b> here a small stand-in decides which knobs are active (median ${Math.round(LT.activity.acc * 100)}% correct). In the plugin this follows directly from the decoded switches.`,
  `<b>Knob pad stays:</b> latent pad to explore, knob pad to fine-tune the result; switching keeps the decoded preset.`,
].map(s => `<li>${s}</li>`).join("");

// ---------- rough preview synth (Web Audio) ----------
let AC = null, S = null, curNote = null, soundOn = false;
function makeSynth() {
  AC = new (window.AudioContext || window.webkitAudioContext)();
  const o = (type) => { const n = AC.createOscillator(); n.type = type; n.start(); return n; };
  const gain = (v = 0) => { const g = AC.createGain(); g.gain.value = v; return g; };
  const tri = o("triangle"), saw = o("sawtooth"), sq = o("square"), sub = o("square"), saw2 = o("sawtooth"), uA = o("sawtooth"), uB = o("sawtooth");
  const gTri = gain(), gSaw = gain(), gSq = gain(), gSub = gain(), gO2 = gain(), gU = gain(), osc1 = gain(1), mix = gain(1);
  const nb = AC.createBuffer(1, AC.sampleRate * 2, AC.sampleRate), nd = nb.getChannelData(0); for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  const noise = AC.createBufferSource(); noise.buffer = nb; noise.loop = true; noise.start(); const gN = gain();
  tri.connect(gTri); saw.connect(gSaw); sq.connect(gSq); [gTri, gSaw, gSq].forEach(g => g.connect(osc1));
  const gO1 = gain(1); osc1.connect(gO1); gO1.connect(mix); sub.connect(gSub); gSub.connect(mix); saw2.connect(gO2); gO2.connect(mix);
  uA.connect(gU); uB.connect(gU); gU.connect(mix); noise.connect(gN); gN.connect(mix);
  const shaper = AC.createWaveShaper(); shaper.oversample = "2x"; mix.connect(shaper);
  const filt = AC.createBiquadFilter(); filt.type = "lowpass"; shaper.connect(filt);
  const amp = gain(0); filt.connect(amp);
  const dry = gain(1), out = gain(0.5); amp.connect(dry); dry.connect(out);
  const dl = AC.createDelay(2.5), dfb = gain(0), dwet = gain(0); amp.connect(dl); dl.connect(dfb); dfb.connect(dl); dl.connect(dwet); dwet.connect(out);
  const conv = AC.createConvolver(), rwet = gain(0); amp.connect(conv); conv.connect(rwet); rwet.connect(out);
  out.connect(AC.destination);
  const lfo = o("sine"), lfoG = gain(0); lfo.connect(lfoG);
  const drift = AC.createBufferSource(); const db = AC.createBuffer(1, AC.sampleRate * 8, AC.sampleRate), dd = db.getChannelData(0);
  let w = 0; for (let i = 0; i < dd.length; i++) { w += (Math.random() * 2 - 1) * 0.002; w *= 0.9995; dd[i] = w; } drift.buffer = db; drift.loop = true; drift.start();
  const driftG = gain(0); drift.connect(driftG);
  const pitched = [tri, saw, sq, sub, saw2, uA, uB];
  pitched.forEach(n => { lfoG.connect(n.detune); driftG.connect(n.detune); });
  S = { tri, saw, sq, sub, saw2, uA, uB, gTri, gSaw, gSq, gSub, gO1, gO2, gU, gN, shaper, filt, amp, dl, dfb, dwet, conv, rwet, lfo, lfoG, driftG, out, pitched, revLen: 0 };
}
const lin = (v, d = 0) => (v == null ? d : v);
function P() {
  const v = state.vals, n = (k, d) => lin(nat(k, v), d), c = (k, d) => { const i = KI[k]; const x = v[i]; return x == null || Number.isNaN(x) ? d : x; };
  const env = (k, d) => 2 ** n(k, d);
  return {
    aA: env("scene/adsr/0/a", -8), aD: env("scene/adsr/0/d", -2), aS: c("scene/adsr/0/s", 1), aR: env("scene/adsr/0/r", -2),
    fA: env("scene/adsr/1/a", -8), fD: env("scene/adsr/1/d", -2), fS: c("scene/adsr/1/s", 1), fR: env("scene/adsr/1/r", -2),
    cutoff: n("scene/filterunit/0/cutoff", 3), res: c("scene/filterunit/0/resonance", 0), envmod: n("scene/filterunit/0/envmod", 0), kt: n("scene/filterunit/0/keytrack", 0),
    shape: n("scene/osc/0/classic/p0", 0), subm: c("scene/osc/0/classic/p3", 0), uni: c("scene/osc/0/classic/p5", 0),
    o1: c("scene/level_o1", 1), o2: c("scene/level_o2", 0), det: n("scene/osc/1/pitch", 0), noise: c("scene/level_noise", 0) * (c("scene/level_noise", 0) > 0.05 ? 1 : 0),
    drive: n("scene/wsunit/drive", 0), glide: c("scene/portamento", 0) > 0.01 ? 2 ** n("scene/portamento", -8) : 0, drift: c("scene/drift", 0),
    lfoHz: 2 ** n("scene/lfo/0/rate", 1), lfoAmt: c("scene/lfo/0/magnitude", 0),
    revT: v[KI["fx/reverb/p3"]] == null || Number.isNaN(v[KI["fx/reverb/p3"]]) ? 0 : 2 ** n("fx/reverb/p3", 0) * (0.5 + c("fx/reverb/p2", 0.5)),
    revMix: Number.isNaN(v[KI["fx/reverb/p9"]]) || v[KI["fx/reverb/p9"]] == null ? 0 : c("fx/reverb/p9", 0) * 0.6,
    dT: Number.isNaN(v[KI["fx/delay/p0"]]) || v[KI["fx/delay/p0"]] == null ? 0 : Math.min(2, 2 ** n("fx/delay/p0", -2)),
    dFb: c("fx/delay/p2", 0) * 0.85, dMix: Number.isNaN(v[KI["fx/delay/p10"]]) || v[KI["fx/delay/p10"]] == null ? 0 : c("fx/delay/p10", 0) * 0.5,
    vol: c("scene/volume", 1),
  };
}
function shaperCurve(db) { const k = Math.max(0, db + 6) / 6, N = 1024, cur = new Float32Array(N);
  for (let i = 0; i < N; i++) { const x = i / (N - 1) * 2 - 1; cur[i] = k < 0.05 ? x : Math.tanh(x * (1 + k * 2)) / Math.tanh(1 + k * 2); } return cur; }
function reverbIR(sec) { const len = Math.max(1, Math.floor(AC.sampleRate * Math.min(6, Math.max(0.1, sec)))), b = AC.createBuffer(2, len, AC.sampleRate);
  for (let ch = 0; ch < 2; ch++) { const d = b.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.5); } return b; }
let lastDrive = null;
function synthUpdate() {
  if (!S) return;
  const p = P(), t = AC.currentTime, sm = 0.03;
  const sh = Math.max(-1, Math.min(1, p.shape));
  S.gTri.gain.setTargetAtTime(Math.max(0, -sh) * 0.9, t, sm); S.gSaw.gain.setTargetAtTime((1 - Math.abs(sh)) * 0.6, t, sm); S.gSq.gain.setTargetAtTime(Math.max(0, sh) * 0.45, t, sm);
  S.gO1.gain.setTargetAtTime(p.o1 ** 2, t, sm); S.gSub.gain.setTargetAtTime(p.subm * 0.4 * p.o1 ** 2, t, sm);
  S.gO2.gain.setTargetAtTime(p.o2 ** 2 * 0.5, t, sm); S.saw2.detune.setTargetAtTime(p.det * 100, t, sm);
  S.gU.gain.setTargetAtTime(p.uni > 0.02 ? 0.3 * p.o1 ** 2 : 0, t, sm); S.uA.detune.setTargetAtTime(p.uni * 60, t, sm); S.uB.detune.setTargetAtTime(-p.uni * 60, t, sm);
  S.gN.gain.setTargetAtTime(p.noise ** 2 * 0.35, t, sm);
  if (lastDrive === null || Math.abs(lastDrive - p.drive) > 0.5) { S.shaper.curve = shaperCurve(p.drive); lastDrive = p.drive; }
  S.filt.Q.setTargetAtTime(0.7 + p.res ** 2 * 18, t, sm);
  S.lfo.frequency.setTargetAtTime(Math.min(30, p.lfoHz), t, sm); S.lfoG.gain.setTargetAtTime(p.lfoAmt * 35, t, sm); S.driftG.gain.setTargetAtTime(p.drift * 2500, t, sm);
  S.dl.delayTime.setTargetAtTime(Math.max(0.01, p.dT), t, 0.08); S.dfb.gain.setTargetAtTime(p.dT ? p.dFb : 0, t, sm); S.dwet.gain.setTargetAtTime(p.dT ? p.dMix : 0, t, sm);
  if (p.revT && Math.abs(S.revLen - p.revT) > 0.25) { S.conv.buffer = reverbIR(p.revT); S.revLen = p.revT; }
  S.rwet.gain.setTargetAtTime(p.revT ? p.revMix : 0, t, sm);
  S.out.gain.setTargetAtTime(0.45 * p.vol, t, sm);
  if (curNote != null) setCutoffStatic(p);
}
const hz = (m) => 440 * 2 ** ((m - 69) / 12);
const cutHz = (p, note, envSemis) => Math.min(18000, Math.max(30, 440 * 2 ** ((p.cutoff + p.kt * (note - 60) + envSemis) / 12)));
function setCutoffStatic(p) { const t = AC.currentTime; S.filt.frequency.setTargetAtTime(cutHz(p, curNote, p.envmod * p.fS), t, 0.05); }
function noteOn(m) {
  if (!S) return; const p = P(), t = AC.currentTime + 0.005, f = hz(m);
  for (const n of S.pitched) { const mult = n === S.sub ? 0.5 : 1; n.frequency.cancelScheduledValues(t);
    if (p.glide && curNote != null) { n.frequency.setValueAtTime(n.frequency.value, t); n.frequency.exponentialRampToValueAtTime(f * mult, t + p.glide); } else n.frequency.setValueAtTime(f * mult, t); }
  const g = S.amp.gain; g.cancelScheduledValues(t); g.setValueAtTime(Math.max(0.0001, g.value), t); g.linearRampToValueAtTime(1, t + p.aA); g.setTargetAtTime(p.aS, t + p.aA, p.aD / 3);
  const fr = S.filt.frequency; fr.cancelScheduledValues(t); fr.setValueAtTime(cutHz(p, m, 0), t);
  fr.exponentialRampToValueAtTime(cutHz(p, m, p.envmod), t + p.fA); fr.setTargetAtTime(cutHz(p, m, p.envmod * p.fS), t + p.fA, p.fD / 3);
  curNote = m;
}
function noteOff() {
  if (!S || curNote == null) return; const p = P(), t = AC.currentTime;
  S.amp.gain.cancelScheduledValues(t); S.amp.gain.setValueAtTime(S.amp.gain.value, t); S.amp.gain.setTargetAtTime(0, t, p.aR / 4);
  S.filt.frequency.cancelScheduledValues(t); S.filt.frequency.setValueAtTime(S.filt.frequency.value, t); S.filt.frequency.setTargetAtTime(cutHz(p, curNote, 0), t, p.fR / 4);
  curNote = null;
}
const soundBtn = document.getElementById("sound");
soundBtn.addEventListener("click", async () => {
  if (!AC) makeSynth();
  soundOn = !soundOn; soundBtn.setAttribute("aria-pressed", soundOn); soundBtn.textContent = soundOn ? "Sound off" : "Sound on";
  if (soundOn) { await AC.resume(); synthUpdate(); } else { noteOff(); AC.suspend(); }
});
const keysEl = document.getElementById("keys"), PCK = "awsedftgyhujk";
let whiteKeyIndex = 0;
for (let m = 48; m <= 84; m++) {
  const b = document.createElement("div"); const black = [1, 3, 6, 8, 10].includes(m % 12);
  b.className = "key" + (black ? " black" : ""); b.tabIndex = 0; b.textContent = m % 12 === 0 ? `C${Math.floor(m / 12) - 1}` : "";
  b.style.left = `${(whiteKeyIndex - (black ? 0.31 : 0)) * 100 / 22}%`;
  if (!black) whiteKeyIndex++;
  b.dataset.m = m; b.setAttribute("role", "button"); b.setAttribute("aria-label", `Note ${m}`);
  const on = async (e) => { e.preventDefault(); if (!soundOn) soundBtn.click(); if (AC.state !== "running") await AC.resume(); noteOn(m); b.classList.add("on"); };
  const off = () => { if (curNote === m) noteOff(); b.classList.remove("on"); };
  b.addEventListener("pointerdown", on); b.addEventListener("pointerup", off); b.addEventListener("pointerleave", off);
  b.addEventListener("pointercancel", off);
  b.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && !e.repeat) on(e); });
  b.addEventListener("keyup", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); off(); } });
  b.addEventListener("blur", off);
  keysEl.appendChild(b);
}
window.addEventListener("keydown", (e) => { if (e.target.matches("input,select,button,textarea") || e.repeat) return; const i = PCK.indexOf(e.key.toLowerCase()); if (i < 0) return;
  keysEl.children[i].dispatchEvent(new PointerEvent("pointerdown")); });
window.addEventListener("keyup", (e) => { const i = PCK.indexOf(e.key.toLowerCase()); if (i < 0) return;
  keysEl.children[i].classList.remove("on"); if (curNote === 48 + i) noteOff(); });

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); render(); });
new MutationObserver(() => { readColors(); render(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
window.addEventListener("blur", () => { noteOff(); keysEl.querySelectorAll(".on").forEach(key => key.classList.remove("on")); });
window.addEventListener("xypad-style-change", () => { readColors(); render(false); });
// Browser-preset contract: retain both editing modes, including structure locks.
function capturePatch() {
  return JSON.parse(JSON.stringify({version:1,model:LT.run,knobCount:NK,label:L[argmax(predict(state.vals))],
    mode:state.mode,base:state.base,values:Array.from(state.vals),axes:{...state.axes},
    latent:{base:lat.base,z:Array.from(lat.z),category:lat.y,axes:{...lat.axes},
      lock:lat.lock?{act:Array.from(lat.lock.act),bools:Array.from(lat.lock.bools),cats:Array.from(lat.lock.cats)}:null}}));
}
function validatePatch(p) {
  const arr=(v,n,check)=>Array.isArray(v)&&v.length===n&&v.every(check);
  const finite=Number.isFinite,bit=n=>n===0||n===1;
  const axes=(a,table)=>a&&table[a.x]&&table[a.y]&&a.x!==a.y&&(a.z===null||(table[a.z]&&a.z!==a.x&&a.z!==a.y));
  const index=(v,n)=>Number.isInteger(v)&&v>=0&&v<n;
  if(!p||p.version!==1||p.model!==LT.run||p.knobCount!==NK||!['latent','knob'].includes(p.mode)
    ||!index(p.base,BASES.length)||!arr(p.values,NK,v=>v===null||(finite(v)&&v>=0&&v<=1))||!axes(p.axes,byName)
    ||!p.latent||!index(p.latent.base,LBASES.length)||!index(p.latent.category,NC)||!arr(p.latent.z,ZD,finite)||!axes(p.latent.axes,AXI))
    throw new Error('This favourite is not compatible with the current model.');
  const lock=p.latent.lock;
  if(lock!==null&&(!lock||!arr(lock.act,NK,bit)||!arr(lock.bools,NB,bit)||!Array.isArray(lock.cats)||lock.cats.length!==CAT_SIZES.length||!lock.cats.every((v,i)=>index(v,CAT_SIZES[i]))))
    throw new Error('This favourite contains an invalid structure lock.');
}
function restorePatch(p) {
  validatePatch(p); // Reject before changing any sound state.
  noteOff();
  state.base=p.base;state.axes={...p.axes};state.vals=p.values.map(v=>v===null?NaN:v);
  lat.base=p.latent.base;lat.z=Float32Array.from(p.latent.z);lat.y=p.latent.category;lat.axes={...p.latent.axes};lat.msg='';
  const lock=p.latent.lock;
  lat.lock=lock?{act:Uint8Array.from(lock.act),bools:Uint8Array.from(lock.bools),cats:[...lock.cats]}:null;
  catSel.value=lat.y;lockBtn.setAttribute('aria-pressed',String(Boolean(lat.lock)));
  latSyncPos();setMode(p.mode);
}
window.XypadPresets={capture:capturePatch,restore:restorePatch,validate:validatePatch};
// start
catSel.value = lat.y; latSyncPos();
setMode(state.mode);

setPadView('orbit');
new ResizeObserver(resize).observe(cv);
window.__xypad = { decodeP, decodeA, measures, LT, lat, AX };     // for checks
})();

