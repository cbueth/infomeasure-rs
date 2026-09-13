// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared helpers for the infomeasure benchmark pages (schema-v2 fragments).
//
// Wrapped in an IIFE so page scripts can destructure from `window.IMV`
// without duplicate top-level `const` declarations.
(function () {

const MEASURE_ORDER = ['entropy', 'mi', 'cmi', 'te', 'cte'];
const MEASURE_LABELS = { entropy: 'Entropy', mi: 'MI', cmi: 'CMI', te: 'TE', cte: 'CTE' };

// Cross-package comparison cells.
const MAIN_APPROACHES = ['discrete', 'ksg', 'kernel_box', 'kernel_gaussian'];
const MAIN_APPROACH_LABELS = {
  discrete: 'Discrete',
  ksg: 'KSG / kNN',
  kernel_box: 'Kernel (box)',
  kernel_gaussian: 'Kernel (Gaussian)',
};

// Detailed (infomeasure) approach tabs, mirroring the historical viewer.
// Ordinal is deliberately last.
const DETAIL_APPROACHES = ['discrete', 'kernel', 'kl', 'renyi', 'tsallis', 'ordinal'];
const DETAIL_APPROACH_LABELS = {
  discrete: 'Discrete',
  kernel: 'Kernel',
  kl: 'KL/KSG/kNN',
  ordinal: 'Ordinal',
  renyi: 'Renyi',
  tsallis: 'Tsallis',
};

const PACKAGE_COLORS = [
  '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00',
  '#a65628', '#f781bf', '#00ced1', '#666666', '#bcbd22',
];
const packageColor = (id) => {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return PACKAGE_COLORS[h % PACKAGE_COLORS.length];
};

// Distinct colors by package order (avoids the hash collisions of packageColor).
function makeColorMap(ids) {
  const m = {};
  ids.forEach((id, i) => { m[id] = PACKAGE_COLORS[i % PACKAGE_COLORS.length]; });
  return m;
}

function fmt(v) {
  if (v == null || Number.isNaN(v)) return '-';
  if (v >= 1) return v.toFixed(3) + 's';
  if (v >= 1e-3) return (v * 1e3).toFixed(1) + 'ms';
  if (v >= 1e-6) return (v * 1e6).toFixed(0) + '\u00b5s';
  return (v * 1e9).toFixed(0) + 'ns';
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

async function fetchJSON(url) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

// Load the package catalog (registry.json) and every available fragment.
// Also fetch the optional alphabet-scaling family (`<id>_alphabet.json`).
async function loadCatalog() {
  const reg = (await fetchJSON('./registry.json')) || { packages: [], excluded: [] };
  const packages = reg.packages || [];
  const fragments = await Promise.all(packages.map((p) => fetchJSON(`./data/${p.id}.json`)));
  const alphabetFrags = await Promise.all(packages.map((p) => fetchJSON(`./data/${p.id}_alphabet.json`)));
  const data = {};
  const alphabet = {};
  packages.forEach((p, i) => {
    if (fragments[i]) data[p.id] = fragments[i];
    if (alphabetFrags[i]) alphabet[p.id] = alphabetFrags[i];
  });
  return { registry: reg, packages, excluded: reg.excluded || [], data, alphabet };
}

function fragmentPackages(frag) {
  return ((frag && frag.meta && frag.meta.packages) || []);
}

function fragmentVersion(frag) {
  const p = fragmentPackages(frag)[0] || {};
  return p.version || '?';
}

function fragmentCollected(frag) {
  const g = frag && frag.meta && frag.meta.generated;
  if (!g) return null;
  const d = new Date(Number(g) * (String(g).length <= 10 ? 1000 : 1));
  return Number.isNaN(d.getTime()) ? null : d.toISOString().slice(0, 10);
}

function fragmentHardware(frag) {
  return (frag && frag.meta && frag.meta.hardware) || null;
}

function meanOf(e) { return e && e.statistics ? e.statistics.mean : null; }
function stdOf(e) { return e && e.statistics ? (e.statistics.stddev ?? null) : null; }

// Representative entries: third-party fragments predate the flag (undefined = yes).
function isRepresentative(e) { return e.representative !== false; }

// Normalize any fragment's approach to a cross-package comparison cell.
function mainApproach(e) {
  const a = e.approach;
  if (a === 'kernel') {
    const kt = (e.params || {}).kernel_type;
    if (kt === 'box') return 'kernel_box';
    if (kt === 'gaussian') return 'kernel_gaussian';
    return null;
  }
  if (MAIN_APPROACHES.includes(a)) return a;
  return null;
}

// Normalize a detailed entry's approach to one of DETAIL_APPROACHES.
// KSG and the KL variants collapse into the "kl" tab, as the old viewer did.
function detailApproach(e) {
  const a = e.approach;
  if (a === 'ksg' || a === 'kl' || a === 'kl_cheb' || a === 'kl_k') return 'kl';
  return a;
}

function limitsOf(pkg, frag) {
  const fp = fragmentPackages(frag)[0] || {};
  return fp.limitations || pkg.limitations || null;
}

// Schema-v2 statistic samples (number of timed iterations pooled).
function samplesOf(e) {
  return e && e.statistics ? (e.statistics.samples ?? null) : null;
}

window.IMV = {
  MEASURE_ORDER, MEASURE_LABELS,
  MAIN_APPROACHES, MAIN_APPROACH_LABELS,
  DETAIL_APPROACHES, DETAIL_APPROACH_LABELS,
  packageColor, makeColorMap, fmt, esc, fetchJSON, loadCatalog,
  fragmentPackages, fragmentVersion, fragmentCollected, fragmentHardware,
  meanOf, stdOf, isRepresentative, mainApproach, detailApproach, limitsOf, samplesOf,
};
})();
