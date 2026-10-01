// Per-frame perception in the browser. Mirrors facial/perception.py and
// facial/features.py (hand tracking) so live samples use the same schema as the
// offline pipeline and the server can compute the same five fields from them.

const FINGERS = { index: [5, 6, 8], middle: [9, 10, 12], ring: [13, 14, 16], pinky: [17, 18, 20] };
const PALM = [0, 5, 9, 13, 17];
const MP_GESTURES = {
  Open_Palm: "open_palm", Closed_Fist: "fist", Pointing_Up: "pointing",
  Thumb_Up: "thumb", Thumb_Down: "thumb", Victory: "two_fingers",
};
export const JITTER_FLOOR = 0.3; // hand-lengths/s of apparent motion on a still hand

const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
const deg = (r) => (r * 180) / Math.PI;

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Hand shape, palm facing and axis from 21 landmarks (see describe_hand in perception.py). */
export function describeHand(pts, world, appearance) {
  const d = (i, j) => Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
  const size = d(9, 0) + 1e-6;
  const extended = {};
  for (const [name, [, pip, tip]] of Object.entries(FINGERS)) extended[name] = d(tip, 0) > 1.15 * d(pip, 0);
  const nExt = Object.values(extended).filter(Boolean).length;
  const thumbOut = d(4, 17) > 1.1 * d(3, 17) && d(4, 5) > 0.5 * size;
  const pinch = d(4, 8) < 0.3 * size;

  let shape;
  if (pinch && !extended.index) shape = "pinch";
  else if (nExt >= 4) shape = "open_palm";
  else if (nExt === 0) shape = thumbOut ? "thumb" : "fist";
  else if (extended.index && nExt === 1) shape = "pointing";
  else if (extended.index && extended.middle && nExt === 2) shape = "two_fingers";
  else shape = "relaxed";

  const ax = pts[9][0] - pts[0][0];
  const ay = pts[9][1] - pts[0][1];
  const axis = Math.abs(ax) > Math.abs(ay) ? "horizontal" : ay < 0 ? "up" : "down";

  // Palm normal points out of the palm toward the camera (-z) for a right-looking hand.
  const sign = appearance === "right" ? 1 : -1;
  const v1 = [pts[5][0] - pts[0][0], pts[5][1] - pts[0][1]];
  const v2 = [pts[17][0] - pts[0][0], pts[17][1] - pts[0][1]];
  const sin2d = ((v1[0] * v2[1] - v1[1] * v2[0]) / (Math.hypot(...v1) * Math.hypot(...v2) + 1e-9)) * sign;
  let n = [0, 0, sin2d];
  if (world) {
    const w = (i) => [world[i].x, world[i].y, world[i].z];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    n = cross(sub(w(5), w(0)), sub(w(17), w(0))).map((x) => x * sign);
  }
  const len = Math.hypot(...n) + 1e-9;
  n = n.map((x) => x / len);
  if (Math.abs(sin2d) > 0.25 && Math.sign(sin2d) !== Math.sign(n[2])) n = [0, 0, Math.sign(sin2d)];
  let facing;
  if (n[2] < -0.6) facing = "palm_out";
  else if (n[2] > 0.6) facing = "back_out";
  else if (Math.abs(n[1]) >= Math.abs(n[0])) facing = n[1] < 0 ? "palm_up" : "palm_down";
  else facing = "palm_side";
  return { shape, facing, axis, size };
}

function faceMetrics(categories) {
  const bs = {};
  for (const c of categories) bs[c.categoryName] = c.score;
  const avg = (...names) => names.reduce((s, n) => s + (bs[n] ?? 0), 0) / names.length;
  const m = {
    smile: avg("mouthSmileLeft", "mouthSmileRight"),
    frown: avg("mouthFrownLeft", "mouthFrownRight"),
    brow_furrow: avg("browDownLeft", "browDownRight"),
    brow_raise: avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    lip_press: avg("mouthPressLeft", "mouthPressRight"),
    blink: avg("eyeBlinkLeft", "eyeBlinkRight"),
    eye_wide: avg("eyeWideLeft", "eyeWideRight"),
    squint: avg("eyeSquintLeft", "eyeSquintRight"),
    jaw: bs.jawOpen ?? 0,
    gaze_side: Math.max(avg("eyeLookOutLeft", "eyeLookInRight"), avg("eyeLookInLeft", "eyeLookOutRight")),
    gaze_down: avg("eyeLookDownLeft", "eyeLookDownRight"),
    // For expressions (expressions.js): cheek raise = genuine smile, sneer = disgust,
    // inner-brow raise = sadness/worry, mouth stretch = fear, lopsided smile = contempt.
    cheek_squint: avg("cheekSquintLeft", "cheekSquintRight"),
    nose_sneer: avg("noseSneerLeft", "noseSneerRight"),
    upper_lip_up: avg("mouthUpperUpLeft", "mouthUpperUpRight"),
    brow_inner_up: bs.browInnerUp ?? 0,
    brow_outer_up: avg("browOuterUpLeft", "browOuterUpRight"),
    mouth_stretch: avg("mouthStretchLeft", "mouthStretchRight"),
    smile_asym: Math.abs((bs.mouthSmileLeft ?? 0) - (bs.mouthSmileRight ?? 0)),
  };
  for (const k of Object.keys(m)) m[k] = round(m[k], 3);
  return m;
}

/** Euler angles matching cv2.RQDecomp3x3 (R = Rz * Ry * Rx), in degrees. */
export function headPose(matrix, columnMajor = true) {
  const at = (r, c) => (columnMajor ? matrix.data[c * 4 + r] : matrix.data[r * 4 + c]);
  return {
    pitch: round(deg(Math.atan2(at(2, 1), at(2, 2))), 1),
    yaw: round(deg(Math.asin(Math.max(-1, Math.min(1, -at(2, 0))))), 1),
    roll: round(deg(Math.atan2(at(1, 0), at(0, 0))), 1),
  };
}

function face(res) {
  if (!res || !res.faceLandmarks || !res.faceLandmarks.length) return null;
  const boxes = res.faceLandmarks.map((lms) => {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of lms) {
      x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
    }
    return [x0, y0, x1, y1];
  });
  let best = 0;
  boxes.forEach((b, i) => {
    const area = (b[2] - b[0]) * (b[3] - b[1]);
    const bb = boxes[best];
    if (area > (bb[2] - bb[0]) * (bb[3] - bb[1])) best = i;
  });
  const [x0, y0, x1, y1] = boxes[best];
  const out = { bbox: [x0, y0, x1, y1].map((v) => round(v, 4)), size: round(y1 - y0, 4) };
  if (res.faceBlendshapes && res.faceBlendshapes[best]) Object.assign(out, faceMetrics(res.faceBlendshapes[best].categories));
  if (res.facialTransformationMatrixes && res.facialTransformationMatrixes[best]) {
    Object.assign(out, headPose(res.facialTransformationMatrixes[best]));
  }
  return out;
}

function hands(res, width, height, mirrored) {
  const out = [];
  if (!res || !res.landmarks) return out;
  const handedness = res.handedness || res.handednesses || [];
  res.landmarks.forEach((lms, i) => {
    const top = (handedness[i] || [])[0] || { categoryName: "Right", score: 0.5 };
    // The label matches how the hand looks in the (non-mirrored) camera frame.
    const appearance = top.categoryName.toLowerCase();
    const side = mirrored ? (appearance === "right" ? "left" : "right") : appearance;
    const pts = lms.map((p) => [p.x * width, p.y * height]);
    const world = res.worldLandmarks && res.worldLandmarks[i] ? res.worldLandmarks[i] : null;
    const geo = describeHand(pts, world, appearance);

    let shape = geo.shape, score = 0.8 * top.score, src = "geometry";
    const g = res.gestures && res.gestures[i] && res.gestures[i][0];
    if (g && MP_GESTURES[g.categoryName] && g.score >= 0.5) {
      shape = MP_GESTURES[g.categoryName]; score = g.score; src = "classifier";
    }
    let ax = 0, ay = 0;
    for (const j of PALM) { ax += pts[j][0]; ay += pts[j][1]; }
    ax /= PALM.length; ay /= PALM.length;
    out.push({
      side, shape, facing: geo.facing, axis: geo.axis, score: round(score, 3), src,
      anchor: [round(ax / width, 4), round(ay / height, 4)], size: round(geo.size / height, 4),
      pts: lms.map((p) => [round(p.x, 4), round(p.y, 4)]),
    });
  });
  return out;
}

function pose(res, width, height) {
  if (!res || !res.landmarks || !res.landmarks.length) return null;
  const lm = res.landmarks[0];
  const P = (i) => [lm[i].x * width, lm[i].y * height, lm[i].visibility ?? 0];
  const ls = P(11), rs = P(12);
  if (Math.min(ls[2], rs[2]) < 0.5) return null;
  const sw = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) + 1e-6;
  const out = { shoulder_w: round(sw / height, 4), tilt: round(deg(Math.atan2(ls[1] - rs[1], Math.abs(ls[0] - rs[0]))), 1) };
  const lw = P(15), rw = P(16);
  if (Math.min(lw[2], rw[2]) >= 0.5) out.arms_open = round(Math.hypot(lw[0] - rw[0], lw[1] - rw[1]) / sw, 3);
  return out;
}

/** One sample in the same schema as run_perception() in perception.py. */
export function buildSample(t, faceRes, handRes, poseRes, width, height, mirrored = false) {
  return {
    t: round(t, 3),
    cut: false,
    faces: faceRes && faceRes.faceLandmarks ? faceRes.faceLandmarks.length : 0,
    face: face(faceRes),
    hands: hands(handRes, width, height, mirrored),
    pose: pose(poseRes, width, height),
  };
}

/** Online version of add_hand_motion(): nearest-neighbour tracks + smoothed speed. */
export class HandTracker {
  constructor(maxGap = 0.35) {
    this.minGap = maxGap;
    this.nextId = 0;
    this.live = []; // {t, anchor, size, id}
    this.recent = new Map(); // id -> last raw speeds
    this.lastT = null;
    this.dt = 0.066;
  }

  update(sample, aspect) {
    const t = sample.t;
    if (this.lastT !== null) this.dt = 0.8 * this.dt + 0.2 * (t - this.lastT);
    this.lastT = t;
    // Slow devices process fewer frames: allow a gap of a few frames before a track ends.
    const maxGap = Math.max(this.minGap, 2.5 * this.dt);
    this.live = this.live.filter((p) => t - p.t <= maxGap);
    const used = new Set();
    for (const h of [...sample.hands].sort((a, b) => b.score - a.score)) {
      let best = null, bestD = 0;
      for (const p of this.live) {
        if (used.has(p.id)) continue;
        const dist = Math.hypot((h.anchor[0] - p.anchor[0]) * aspect, h.anchor[1] - p.anchor[1]);
        if (dist < 2.5 * Math.max(h.size, p.size) && (best === null || dist < bestD)) { best = p; bestD = dist; }
      }
      let raw = 0;
      if (best === null) h.track = this.nextId++;
      else {
        h.track = best.id;
        raw = bestD / Math.max(h.size, 1e-3) / Math.max(t - best.t, 1e-3);
      }
      const hist = (this.recent.get(h.track) || []).concat(raw).slice(-3);
      this.recent.set(h.track, hist);
      h.speed = round(Math.max(0, hist.reduce((a, b) => a + b, 0) / hist.length - JITTER_FLOOR), 3);
      used.add(h.track);
    }
    this.live = this.live.filter((p) => !used.has(p.id))
      .concat(sample.hands.map((h) => ({ t, anchor: h.anchor, size: h.size, id: h.track })));
  }
}
