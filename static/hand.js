// The cartoon glove hand used by the opening animation, the logo and the mascots.
// Geometry lives in a 200 x 240 box; the wrist pivot is at (100, 214).

export const INK = "#1e1b2e";
export const PAPER = "#fff6e6";
export const COLORS = { blue: "#5b6cff", yellow: "#ffc83d", coral: "#ff6b6b", mint: "#33d1a0", sky: "#7fd3ff", orange: "#ff9f43" };
export const PIVOT = [100, 214];
export const CENTER = [100, 130];

// Drawing order doubles as the order the pencil sketches them in.
export const SHAPES = [
  { kind: "capsule", base: [68, 168], tip: [30, 128], r: 12, fill: "#ffffff" }, // thumb
  { kind: "capsule", base: [72, 124], tip: [64, 50], r: 12, fill: "#ffffff" }, // index
  { kind: "capsule", base: [97, 120], tip: [97, 38], r: 12.5, fill: "#ffffff" }, // middle
  { kind: "capsule", base: [121, 124], tip: [128, 50], r: 12, fill: "#ffffff" }, // ring
  { kind: "capsule", base: [140, 134], tip: [154, 80], r: 10.5, fill: "#ffffff" }, // pinky
  { kind: "rrect", x: 54, y: 110, w: 92, h: 86, r: 36, fill: "#ffffff" }, // palm
  { kind: "rrect", x: 64, y: 188, w: 72, h: 32, r: 11, fill: COLORS.yellow }, // cuff
  { kind: "line", pts: [[84, 150], [84, 172]] }, // glove stitches
  { kind: "line", pts: [[100, 146], [100, 174]] },
  { kind: "line", pts: [[116, 150], [116, 172]] },
];

function arc(cx, cy, r, a0, a1, n) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}

function segment(a, b, n) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push([a[0] + ((b[0] - a[0]) * i) / n, a[1] + ((b[1] - a[1]) * i) / n]);
  return out;
}

/** Closed (or open, for lines) outline of a shape as a dense point list. */
export function outline(s) {
  if (s.kind === "line") return segment(s.pts[0], s.pts[1], 6);
  if (s.kind === "capsule") {
    const [bx, by] = s.base, [tx, ty] = s.tip;
    const ang = Math.atan2(ty - by, tx - bx);
    const n = [Math.cos(ang + Math.PI / 2) * s.r, Math.sin(ang + Math.PI / 2) * s.r];
    return [
      ...segment([bx + n[0], by + n[1]], [tx + n[0], ty + n[1]], 8),
      ...arc(tx, ty, s.r, ang + Math.PI / 2, ang - Math.PI / 2, 12),
      ...segment([tx - n[0], ty - n[1]], [bx - n[0], by - n[1]], 8),
      ...arc(bx, by, s.r, ang - Math.PI / 2, ang - (3 * Math.PI) / 2, 12),
    ];
  }
  const { x, y, w, h, r } = s;
  return [
    ...segment([x + r, y], [x + w - r, y], 6), ...arc(x + w - r, y + r, r, -Math.PI / 2, 0, 8),
    ...segment([x + w, y + r], [x + w, y + h - r], 6), ...arc(x + w - r, y + h - r, r, 0, Math.PI / 2, 8),
    ...segment([x + w - r, y + h], [x + r, y + h], 6), ...arc(x + r, y + h - r, r, Math.PI / 2, Math.PI, 8),
    ...segment([x, y + h - r], [x, y + r], 6), ...arc(x + r, y + r, r, Math.PI, (3 * Math.PI) / 2, 8),
  ];
}

function svgPath(s) {
  const pts = outline(s);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join("");
  return s.kind === "line" ? d : `${d}Z`;
}

/** Inline SVG of the hand; `cls` lets CSS animate it (e.g. a stepped wave). */
export function handSvg({ size = 64, cls = "", label = "" } = {}) {
  const parts = SHAPES.map((s) => s.kind === "line"
    ? `<path d="${svgPath(s)}" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round"/>`
    : `<path d="${svgPath(s)}" fill="${s.fill}" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/>`).join("");
  const a11y = label ? `role="img" aria-label="${label}"` : 'aria-hidden="true"';
  return `<svg class="${cls}" ${a11y} width="${size}" height="${Math.round(size * 1.2)}" viewBox="0 0 200 240">`
    + `<g class="hand-g" style="transform-origin:${PIVOT[0]}px ${PIVOT[1]}px">${parts}</g></svg>`;
}
