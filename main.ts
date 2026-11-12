// pumpfun-top5-forecast-01 entry (Node 22, standard library only).
// A Monte-Carlo forecaster over the house's 10-minute snapshot history.
//
// Model (everything is derived from the history read at run time; no
// hard-coded times, coins or caps):
// - Each coin's log10 market cap (in SOL) is a jump-diffusion: drift from
//   its own recent momentum, decayed with age (young coins run hot, old
//   coins bleed), plus a long-run negative drift; Gaussian diffusion; and
//   a per-hour crash hazard (a memecoin either sustains or collapses —
//   the practice history shows top coins vanishing from the rows within
//   10 minutes, while sustainers keep adding ~+0.06 log per 10 min).
// - Newcomers (coins born after the history ends) are simulated too:
//   their read-time caps come from a Pareto tail fitted to newborn peak
//   caps in the history (Hill estimator), with the count from the
//   history's own launch rate. They cannot be named (the scorer drops
//   rows for mints the history never names), so their probability mass
//   correctly stays in each place's unnamed share. In the practice, 3 of
//   the 5 places at the last read went to coins born after the cut.
// - The one forecast is scored at every answer read, so every probability
//   is the average over the seven reads of the simulated rank
//   distribution. Sizes are the simulated distribution of the cap at each
//   place, averaged over the reads.
// - Deterministic: one fixed seed, no wall clock, no Math.random, so two
//   runs on the same history are byte-identical.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// ---- tunables (calibrated on the kit's practice history; see run record) ----
const SEED = 20261112;
const NSIM = 3000; // Monte-Carlo paths
const TAU_H = 1.5; // momentum decay with age, hours
const MU_BASE = -0.1; // long-run drift of log10 cap, per hour
const MOM_SHRINK = 0.3; // shrink of a coin's own recent hourly drift
const MOM_CLAMP_LO = -0.3;
const MOM_CLAMP_HI = 0.6;
const SIGMA_YOUNG = 0.35; // per-hour sd of log10 cap, age < 1h
const SIGMA_OLD = 0.28; // per-hour sd, older coins
const CRASH_CURVE = 0.35; // per-hour crash hazard, bonding-curve coins
const CRASH_POOL = 0.25; // per-hour crash hazard, graduated (pool) coins
const CRASH_LO = 0.7; // crash size in log10, uniform
const CRASH_HI = 1.5;
const MIN_P = 0.002; // smallest probability worth naming
const TAIL_FRAC = 0.02; // newborn peaks above the tail threshold
const NEWCOMER_CAP = 5.0; // sanity ceiling on a newcomer read-time log-cap
const NEWCOMER_SCALE = 1; // scales the expected tail-newcomer count

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}
interface Mark {
  at: string;
  ok: boolean;
  counts: { eligible: number };
  rows: Row[] | null;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Standard normal from uniform draws (Box-Muller, no cached spare: deterministic). */
function randn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
/** Poisson sample (Knuth). */
function poisson(rng: () => number, mean: number): number {
  if (mean <= 0) return 0;
  const L = Math.exp(-mean);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rng();
  } while (p > L);
  return k - 1;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  chart_size: number;
  reads: string[];
  size: { edges: number[] };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const lastT = Date.parse(lastMark.at) / 1000;
const readTimes = task.reads.map((r) => Date.parse(r) / 1000);
const readHours = readTimes.map((t) => Math.max(0, (t - lastT) / 3600));
const maxH = Math.max(...readHours);

const sizeOf = (lamports: string | number): number =>
  Math.log10(Number(BigInt(lamports))) - 9;

// ---- per-coin series ----
interface Coin {
  mint: string;
  x0: number; // log10 cap in SOL at the last mark
  created: number; // seconds
  venue: string;
  complete: boolean;
  muMom: number; // shrunk recent hourly drift, per hour
  sigma: number; // per-hour sd
  crash: number; // per-hour crash hazard
  lastRank: number;
}
const byMint = new Map<string, { created: number; obs: Array<{ t: number; x: number; venue: string; complete: boolean; rank: number }> }>();
for (const m of marks) {
  const t = Date.parse(m.at) / 1000;
  for (const r of m.rows!) {
    const e = byMint.get(r.mint) ?? { created: Date.parse(r.created_at) / 1000, obs: [] };
    e.obs.push({ t, x: sizeOf(r.market_cap_lamports), venue: r.venue, complete: r.complete, rank: r.rank });
    byMint.set(r.mint, e);
  }
}
// Newborn peak caps: a coin's max log-cap within 2h of its birth.
const newbornPeaks: number[] = [];
for (const e of byMint.values()) {
  let peak = -Infinity;
  for (const o of e.obs) if (o.t - e.created <= 2 * 3600 && o.x > peak) peak = o.x;
  if (peak > -Infinity) newbornPeaks.push(peak);
}
newbornPeaks.sort((a, b) => b - a);
// Pareto tail of newborn peaks: threshold at the (1-TAIL_FRAC) quantile,
// Hill estimator for alpha; fall back to the empirical top-24.
const tailN = Math.max(5, Math.floor(newbornPeaks.length * TAIL_FRAC));
const tailPeaks = newbornPeaks.slice(0, tailN);
const xminLog = tailPeaks.length > 0 ? tailPeaks[tailPeaks.length - 1]! : 3.0;
// Hill estimator in log space: with x = 10^p, sum(ln(x/xmin)) =
// ln(10) * sum(p - xminLog), so alpha = 1 + n / (ln10 * sum(p - xminLog)).
let alpha = 3.5;
{
  const s = tailPeaks.reduce((a, p) => a + (p - xminLog), 0);
  if (tailPeaks.length >= 5 && s > 0) alpha = Math.min(6, Math.max(1.5, 1 + tailPeaks.length / (s * Math.LN10)));
}
// Launch rate (coins/hour) from the eligible counts of the last two marks.
let launchRate = 1000;
if (marks.length >= 2) {
  const a = marks[marks.length - 2]!;
  const b = lastMark;
  const dtH = (Date.parse(b.at) - Date.parse(a.at)) / 3600000;
  const dElig = (b.counts?.eligible ?? 0) - (a.counts?.eligible ?? 0);
  if (dtH > 0 && dElig > 0) launchRate = dElig / dtH;
}
// Expected tail newcomers (peak above xmin) born in the 2h before a read.
const newcomerMean = Math.max(1, launchRate * 2 * TAIL_FRAC) * NEWCOMER_SCALE;

// Candidates: every coin in the last mark's rows, plus any coin that was
// ever top-50 (it may have dipped out of the last chart but can rebound).
const candMints = new Set<string>();
for (const r of lastMark.rows!) candMints.add(r.mint);
for (const m of marks) for (const r of m.rows!) if (r.rank <= 50) candMints.add(r.mint);

const coins: Coin[] = [];
for (const mint of candMints) {
  const e = byMint.get(mint)!;
  const obs = [...e.obs].sort((a, b) => a.t - b.t);
  const last = obs[obs.length - 1]!;
  const ageH = Math.max(0, (lastT - e.created) / 3600);
  // Recent hourly drift: change over the last hour of observations
  // (or the whole observed window when shorter), shrunk and clamped.
  const tBack = Math.max(obs[0]!.t, lastT - 3600);
  let base = obs[0]!;
  for (const o of obs) if (o.t <= tBack + 1) base = o;
  const spanH = Math.max((last.t - base.t) / 3600, 1 / 6);
  const rawDrift = (last.x - base.x) / spanH;
  const muMom = Math.min(MOM_CLAMP_HI, Math.max(MOM_CLAMP_LO, MOM_SHRINK * rawDrift));
  const sigma = ageH < 1 ? SIGMA_YOUNG : SIGMA_OLD;
  coins.push({
    mint,
    x0: last.x,
    created: e.created,
    venue: last.venue,
    complete: last.complete,
    muMom,
    sigma,
    crash: last.venue === 'pool' ? CRASH_POOL : CRASH_CURVE,
    lastRank: last.rank,
  });
}
if (coins.length < task.places) throw new Error(`only ${coins.length} candidate coins`);

// ---- Monte Carlo ----
// Drift at future time s (hours after the last mark) for a coin of age
// ageH at the last mark: momentum decays as the coin ages, plus base drift.
const driftAt = (c: Coin, ageH: number, s: number): number =>
  c.muMom * Math.exp(-(ageH + s) / TAU_H) + MU_BASE;

const rng = mulberry32(SEED);
const placeCounts = new Map<string, number[]>(); // mint -> counts per place
const sizeCounts: number[][] = Array.from({ length: task.places }, () =>
  Array.from({ length: task.size.edges.length + 1 }, () => 0),
);
const edges = task.size.edges;
const binOf = (x: number): number => {
  if (x < edges[0]!) return 0;
  for (let i = 0; i < edges.length; i++) if (x < edges[i]!) return i;
  return edges.length;
};

const ageH = coins.map((c) => Math.max(0, (lastT - c.created) / 3600));
const nSteps = Math.max(1, Math.ceil(maxH));
const stepHours = Array.from({ length: nSteps }, (_, i) => (i + 1) * (maxH / nSteps));
const readStep = readHours.map((h) => {
  let s = 0;
  while (s < nSteps - 1 && stepHours[s]! < h) s++;
  return s;
});

for (let sim = 0; sim < NSIM; sim++) {
  // Simulate each candidate's log-cap at every step.
  const paths: number[][] = coins.map(() => new Array<number>(nSteps).fill(-Infinity));
  const alive: boolean[] = coins.map(() => true);
  for (let i = 0; i < coins.length; i++) {
    const c = coins[i]!;
    let x = c.x0;
    let prevH = 0;
    for (let s = 0; s < nSteps; s++) {
      const h = stepHours[s]!;
      const dt = h - prevH;
      prevH = h;
      if (!alive[i]) {
        paths[i]![s] = -Infinity;
        continue;
      }
      // Crash hazard over this step: the coin collapses out of contention.
      if (rng() < 1 - Math.exp(-c.crash * dt)) {
        alive[i] = false;
        paths[i]![s] = -Infinity;
        continue;
      }
      // Drift integrated over the step (midpoint rule) + diffusion.
      const mid = h - dt / 2;
      const mu = driftAt(c, ageH[i]!, mid);
      x += mu * dt + c.sigma * Math.sqrt(dt) * randn(rng);
      paths[i]![s] = x;
    }
  }
  // Newcomers per read: Poisson count of tail newborns, caps from the
  // Pareto tail above xminLog (capped for sanity).
  const newcomers: number[][] = readHours.map(() => []);
  for (let ri = 0; ri < readHours.length; ri++) {
    const k = poisson(rng, newcomerMean);
    for (let j = 0; j < k; j++) {
      const u = Math.max(rng(), 1e-9);
      const x = Math.min(NEWCOMER_CAP, xminLog + (-Math.log(u) / (alpha * Math.LN10)));
      newcomers[ri]!.push(x);
    }
  }
  // Rank at each read.
  for (let ri = 0; ri < readHours.length; ri++) {
    const s = readStep[ri]!;
    const scored: Array<{ mint: string | null; x: number }> = [];
    for (let i = 0; i < coins.length; i++) {
      const x = paths[i]![s]!;
      if (x > -Infinity) scored.push({ mint: coins[i]!.mint, x });
    }
    for (const x of newcomers[ri]!) scored.push({ mint: null, x });
    scored.sort((a, b) => b.x - a.x);
    for (let k = 0; k < task.places; k++) {
      const row = scored[k];
      if (!row) break;
      sizeCounts[k]![binOf(row.x)]!++;
      if (row.mint !== null) {
        const arr = placeCounts.get(row.mint) ?? Array.from({ length: task.places }, () => 0);
        arr[k]!++;
        placeCounts.set(row.mint, arr);
      }
    }
  }
}

const totalReads = NSIM * readHours.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (const [mint, arr] of placeCounts) {
  for (let k = 0; k < task.places; k++) {
    const q = arr[k]! / totalReads;
    if (q >= MIN_P) {
      probabilities.push({ mint, place: k + 1, p: Math.floor(q * MICRO) / MICRO });
    }
  }
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || (a.mint < b.mint ? -1 : 1));

const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < task.places; k++) {
  for (let bin = 0; bin <= edges.length; bin++) {
    const q = sizeCounts[k]![bin]! / totalReads;
    if (q > 0) sizes.push({ place: k + 1, bin, p: Math.floor(q * MICRO) / MICRO });
  }
}

// top5: best mint per place, distinct.
const top5: string[] = [];
for (let k = 0; k < task.places; k++) {
  let best: string | null = null;
  let bestQ = -1;
  for (const [mint, arr] of placeCounts) {
    if (top5.includes(mint)) continue;
    if (arr[k]! > bestQ) {
      bestQ = arr[k]!;
      best = mint;
    }
  }
  if (best !== null) top5.push(best);
}
if (top5.length < task.places) {
  for (const c of [...coins].sort((a, b) => a.lastRank - b.lastRank)) {
    if (top5.length >= task.places) break;
    if (!top5.includes(c.mint)) top5.push(c.mint);
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
