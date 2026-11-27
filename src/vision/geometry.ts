// 多视几何：内参、8 点法 RANSAC 基础矩阵、本质矩阵分解、三角化、DLT-PnP、闭环相似变换
import {
  mat3Mul, mat3Transpose, mat3Vec, normalize3, shuffle, svd, v3, type Mat3,
} from '../core/linalg';
import { Rng } from '../core/rng';
import type { CameraPose, Vec3 } from '../core/types';

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  K: Mat3;
}

export function intrinsics(w: number, h: number, fovYDeg: number): Intrinsics {
  const fy = h / (2 * Math.tan((fovYDeg * Math.PI) / 360));
  const fx = fy; // 方形像素
  const K = Float32Array.of(fx, 0, w / 2, 0, fy, h / 2, 0, 0, 1);
  return { fx, fy, cx: w / 2, cy: h / 2, K };
}

export interface Norm2D {
  pts: Float64Array; // n*2
  T: Float64Array;   // 3x3 归一化变换
}

function normalize2d(pts: { x: number; y: number }[]): Norm2D {
  const n = pts.length;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let d = 0;
  for (const p of pts) d += Math.hypot(p.x - mx, p.y - my);
  d /= n;
  const s = d > 1e-9 ? Math.SQRT2 / d : 1;
  const T = Float64Array.of(s, 0, -s * mx, 0, s, -s * my, 0, 0, 1);
  const out = new Float64Array(n * 2);
  pts.forEach((p, i) => {
    out[i * 2] = s * (p.x - mx);
    out[i * 2 + 1] = s * (p.y - my);
  });
  return { pts: out, T };
}

function mat3To64(m: Mat3): Float64Array {
  return Float64Array.from(m);
}

/** 8 点法（归一化坐标输入），返回 3x3 基础矩阵（原始像素坐标下） */
function fundamentalFromNorm(normA: Norm2D, normB: Norm2D, TA: Float64Array, TB: Float64Array): Mat3 {
  const n = normA.pts.length / 2;
  const A = new Float64Array(n * 9);
  for (let i = 0; i < n; i++) {
    const x = normA.pts[i * 2], y = normA.pts[i * 2 + 1];
    const xp = normB.pts[i * 2], yp = normB.pts[i * 2 + 1];
    A.set([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1], i * 9);
  }
  // A 的零空间：AtA 最小特征向量
  const AtA = new Float64Array(9 * 9);
  for (let i = 0; i < 9; i++)
    for (let j = 0; j < 9; j++)
      for (let k = 0; k < n; k++) AtA[i * 9 + j] += A[k * 9 + i] * A[k * 9 + j];
  const { Vt } = svd(AtA, 9, 9);
  const f = Vt.slice(8 * 9, 9 * 9); // 最后一行 = 最小奇异值对应向量
  let F: Float64Array = Float64Array.of(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
  // 秩 2 约束
  const fSvd = svd(F, 3, 3);
  const S = fSvd.S;
  S[2] = 0;
  const Sm = Float64Array.of(S[0], 0, 0, 0, S[1], 0, 0, 0, 0);
  // F = U * S * Vt，按 mul64(a, b, ar, ac, bc) 分步计算
  const US = mul64(fSvd.U, Sm, 3, 3, 3);
  F = mul64(US, fSvd.Vt, 3, 3, 3);
  // 反归一化：F = TB^T F_norm TA
  const TBt = transpose(TB, 3, 3);
  const result = mul64(mul64(TBt, F, 3, 3, 3), TA, 3, 3, 3);
  return Float32Array.from(result);
}

function mul64(a: Float64Array, b: Float64Array, ar: number, ac: number, bc: number): Float64Array {
  const br = ac;
  const o = new Float64Array(ar * bc);
  for (let i = 0; i < ar; i++)
    for (let j = 0; j < bc; j++)
      for (let k = 0; k < ac; k++) o[i * bc + j] += a[i * ac + k] * b[k * bc + j];
  void br;
  return o;
}
function transpose(a: Float64Array, n: number, m: number): Float64Array {
  const o = new Float64Array(m * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) o[j * n + i] = a[i * m + j];
  return o;
}

export interface RansacResult {
  model: Mat3;
  inliers: number[];
  inlierMask: Uint8Array;
}

/** RANSAC 基础矩阵 */
export function ransacFundamental(
  ptsA: { x: number; y: number }[],
  ptsB: { x: number; y: number }[],
  seed = 42,
  threshold = 1.2,
  iterations = 300,
): RansacResult | null {
  if (ptsA.length < 8) return null;
  const rng = new Rng(seed);
  const n = ptsA.length;
  let best: RansacResult | null = null;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let it = 0; it < iterations; it++) {
    const sample = shuffle(idx, rng.next.bind(rng)).slice(0, 8);
    const nA = normalize2d(sample.map((i) => ptsA[i]));
    const nB = normalize2d(sample.map((i) => ptsB[i]));
    let F: Mat3;
    try {
      F = fundamentalFromNorm(nA, nB, nA.T, nB.T);
    } catch {
      continue;
    }
    const inliers: number[] = [];
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (sampsonDistance(F, ptsA[i], ptsB[i]) < threshold) {
        inliers.push(i);
        mask[i] = 1;
      }
    }
    if (!best || inliers.length > best.inliers.length) {
      best = { model: F, inliers, inlierMask: mask };
    }
  }
  if (best && best.inliers.length >= 8) {
    // 用全部内点重新精算
    const nA = normalize2d(best.inliers.map((i) => ptsA[i]));
    const nB = normalize2d(best.inliers.map((i) => ptsB[i]));
    best.model = fundamentalFromNorm(nA, nB, nA.T, nB.T);
  }
  return best;
}

/** Sampson 近似重投影误差（像素） */
function sampsonDistance(F: Mat3, a: { x: number; y: number }, b: { x: number; y: number }): number {
  const pa = Float64Array.of(a.x, a.y, 1);
  const pb = Float64Array.of(b.x, b.y, 1);
  const Fp = mulMatVec64(F, pa);
  const Ftp = mulMatVec64(mat3Transpose(F), Float32Array.from(pb));
  const num = pb[0] * Fp[0] + pb[1] * Fp[1] + pb[2] * Fp[2];
  const den = Fp[0] ** 2 + Fp[1] ** 2 + Ftp[0] ** 2 + Ftp[1] ** 2 + 1e-9;
  return (num * num) / den;
}

function mulMatVec64(m: Mat3 | Float64Array, v: Float64Array | Float32Array): Float64Array {
  return Float64Array.of(
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  );
}

// ---------- 本质矩阵与相对位姿 ----------

export interface RelativePose {
  R: Mat3;
  t: Vec3; // 单位平移（方向），尺度未知
}

export function essentialFromFundamental(F: Mat3, K: Mat3): Mat3 {
  const Kt = mat3Transpose(K);
  const E64 = mul64(mul64(mat3To64(Kt), mat3To64(F), 3, 3, 3), mat3To64(K), 3, 3, 3);
  // 强制本质矩阵奇异值 (1,1,0)
  const { U, Vt } = svd(E64, 3, 3);
  const Sm = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 0);
  return Float32Array.from(mul64(mul64(U, Sm, 3, 3, 3), Vt, 3, 3, 3));
}

/** 本质矩阵四解，用三角化 cheirality 选正确解 */
export function decomposeEssential(
  E: Mat3, K: Mat3,
  ptsA: { x: number; y: number }[],
  ptsB: { x: number; y: number }[],
): RelativePose | null {
  const { U, Vt } = svd(mat3To64(E), 3, 3);
  const W = Float64Array.of(0, -1, 0, 1, 0, 0, 0, 0, 1);
  const Ut = transpose(U, 3, 3);
  const R1 = Float32Array.from(mul64(mul64(U, W, 3, 3, 3), Vt, 3, 3, 3));
  const R2 = Float32Array.from(mul64(mul64(U, transpose(W, 3, 3), 3, 3, 3), Vt, 3, 3, 3));
  const t3 = Float64Array.of(U[2], U[5], U[8]); // U 第三列
  const candidates: RelativePose[] = [
    { R: ensureDet(R1), t: v3(t3[0], t3[1], t3[2]) },
    { R: ensureDet(R1), t: v3(-t3[0], -t3[1], -t3[2]) },
    { R: ensureDet(R2), t: v3(t3[0], t3[1], t3[2]) },
    { R: ensureDet(R2), t: v3(-t3[0], -t3[1], -t3[2]) },
  ];
  void Ut;
  let best: { pose: RelativePose; good: number } | null = null;
  for (const pose of candidates) {
    let good = 0;
    for (let i = 0; i < Math.min(ptsA.length, 40); i++) {
      const X = triangulate(K, identityPose(), toSecondPose(pose), ptsA[i], ptsB[i]);
      if (!X || !isFinite(X.x)) continue;
      const x1 = X; // 相机 A 在原点看 -Z，前方点 z<0
      const x2 = add3(mat3Vec(pose.R, X), pose.t);
      if (x1.z < 0 && x2.z < 0) good++;
    }
    if (!best || good > best.good) best = { pose, good };
  }
  return best && best.good > 0 ? best.pose : null;
}

function ensureDet(R: Mat3): Mat3 {
  const det = R[0] * (R[4] * R[8] - R[5] * R[7]) -
    R[1] * (R[3] * R[8] - R[5] * R[6]) +
    R[2] * (R[3] * R[7] - R[4] * R[6]);
  if (det < 0) return Float32Array.from(R.map((v) => -v));
  return R;
}

const add3 = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });

function identityPose(): CameraPose {
  return {
    index: 0, R: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
    t: v3(), position: v3(), scale: 1,
  };
}

function toSecondPose(rp: RelativePose): CameraPose {
  const Rt = mat3Transpose(rp.R);
  const pos = scale3(mat3Vec(Rt, rp.t), -1);
  return { index: 1, R: rp.R, t: rp.t, position: pos, scale: 1 };
}
const scale3 = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });

/** 线性三角化（DLT），输入两帧位姿与像素坐标，返回世界系下的 3D 点。
 *  相机看向 -Z：相机坐标 Xc = R X + t，投影 u = fx * Xc.x/(-Xc.z) + cx。 */
export function triangulate(
  K: Mat3, poseA: CameraPose, poseB: CameraPose,
  pxA: { x: number; y: number }, pxB: { x: number; y: number },
): Vec3 | null {
  const rows: number[] = [];
  for (const [pose, px] of [[poseA, pxA], [poseB, pxB]] as const) {
    const R = pose.R;
    const t = pose.t;
    // 相机坐标三行
    const r1 = [R[0], R[1], R[2], t.x];
    const r2 = [R[3], R[4], R[5], t.y];
    const r3 = [R[6], R[7], R[8], t.z];
    // u = fx * xc/(-zc) + cx  =>  (u-cx)*(-zc) = fx*xc
    //   => fx*r1·X + (u-cx)*r3·X = 0
    //   fy*r2·X + (v-cy)*r3·X = 0
    const { fx, fy, cx, cy } = mat3ToIntrinsics(K);
    rows.push(
      fx * r1[0] + (px.x - cx) * r3[0],
      fx * r1[1] + (px.x - cx) * r3[1],
      fx * r1[2] + (px.x - cx) * r3[2],
      fx * r1[3] + (px.x - cx) * r3[3],
    );
    rows.push(
      fy * r2[0] + (px.y - cy) * r3[0],
      fy * r2[1] + (px.y - cy) * r3[1],
      fy * r2[2] + (px.y - cy) * r3[2],
      fy * r2[3] + (px.y - cy) * r3[3],
    );
  }
  const A = Float64Array.from(rows);
  const AtA = new Float64Array(4 * 4);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) AtA[i * 4 + j] += A[k * 4 + i] * A[k * 4 + j];
  let Vt: Float64Array;
  try {
    ({ Vt } = svd(AtA, 4, 4));
  } catch {
    return null;
  }
  const nullVec = Vt.slice(3 * 4, 4 * 4);
  if (Math.abs(nullVec[3]) < 1e-10) return null;
  return { x: nullVec[0] / nullVec[3], y: nullVec[1] / nullVec[3], z: nullVec[2] / nullVec[3] };
}

function mat3ToIntrinsics(K: Mat3): { fx: number; fy: number; cx: number; cy: number } {
  return { fx: K[0], fy: K[4], cx: K[2], cy: K[5] };
}

// ---------- PnP（DLT + RANSAC） ----------

export function pnpRansac(
  K: Mat3, worldPts: Vec3[], imgPts: { x: number; y: number }[], seed = 7,
): CameraPose | null {
  if (worldPts.length < 6) return null;
  const rng = new Rng(seed);
  const n = worldPts.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  let best: { pose: CameraPose; inliers: number } | null = null;
  for (let it = 0; it < 120; it++) {
    const sample = shuffle(idx, rng.next.bind(rng)).slice(0, Math.min(6, n));
    const pose = dltPnp(K, sample.map((i) => worldPts[i]), sample.map((i) => imgPts[i]));
    if (!pose) continue;
    let inl = 0;
    for (let i = 0; i < n; i++) {
      if (reprojectionError(K, pose, worldPts[i], imgPts[i]) < 2.0) inl++;
    }
    if (!best || inl > best.inliers) best = { pose, inliers: inl };
  }
  if (!best) return null;
  // 全部内点精算
  const inlierIdx = idx.filter((i) => reprojectionError(K, best!.pose, worldPts[i], imgPts[i]) < 2.0);
  if (inlierIdx.length >= 6) {
    const refined = dltPnp(K, inlierIdx.map((i) => worldPts[i]), inlierIdx.map((i) => imgPts[i]));
    if (refined) return refined;
  }
  return best.pose;
}

function dltPnp(K: Mat3, worldPts: Vec3[], imgPts: { x: number; y: number }[]): CameraPose | null {
  if (worldPts.length < 6) return null;
  const Kinv = invert3(K);
  // 归一化相机射线（相机看向 -Z）：x_cam = K^-1 [u,v,1]，第三分量取负表示前方
  const A = new Float64Array(worldPts.length * 2 * 12);
  worldPts.forEach((X, i) => {
    const nh = mulMatVec64(Kinv, Float32Array.of(imgPts[i].x, imgPts[i].y, 1));
    const x = nh[0];
    const y = nh[1];
    // xc = r1·X+t1, yc = r2·X+t2, zc = r3·X+t3；投影 x = xc/(-zc)
    // => x*r3·X + r1·X = 0（齐次 M=[R|t]，12 个未知数按行展开）
    const row0 = [
      X.x, X.y, X.z, 1, 0, 0, 0, 0, x * X.x, x * X.y, x * X.z, x,
    ];
    const row1 = [
      0, 0, 0, 0, X.x, X.y, X.z, 1, y * X.x, y * X.y, y * X.z, y,
    ];
    A.set(row0, i * 24);
    A.set(row1, i * 24 + 12);
  });
  const AtA = new Float64Array(12 * 12);
  for (let i = 0; i < 12; i++)
    for (let j = 0; j < 12; j++)
      for (let k = 0; k < worldPts.length * 2; k++) AtA[i * 12 + j] += A[k * 12 + i] * A[k * 12 + j];
  let Vt: Float64Array;
  try {
    ({ Vt } = svd(AtA, 12, 12));
  } catch {
    return null;
  }
  const p = Vt.slice(11 * 12, 12 * 12);
  const M = Float64Array.of(...p);
  const Rraw = Float64Array.of(M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]);
  const { U, Vt: VtR } = svd(Rraw, 3, 3);
  let R = mul64(U, VtR, 3, 3, 3);
  if (det3(R) < 0) R = R.map((v) => -v);
  const scale = Math.cbrt(det3(mul64(transpose(R, 3, 3), Rraw, 3, 3, 3)) || 1);
  let t = v3(M[3] / scale, M[7] / scale, M[11] / scale);
  const Rf = Float32Array.from(R);
  // cheirality：多数世界点应落在相机前方（zc<0），否则整体翻转
  let frontCount = 0;
  for (const X of worldPts) {
    const zc = Rf[6] * X.x + Rf[7] * X.y + Rf[8] * X.z + t.z;
    if (zc < 0) frontCount++;
  }
  if (frontCount < worldPts.length / 2) {
    R = R.map((v) => -v);
    t = scale3(t, -1);
  }
  const Rfinal = Float32Array.from(R);
  const Rt = mat3Transpose(Rfinal);
  const position = scale3(mat3Vec(Rt, t), -1);
  return { index: -1, R: Rfinal, t, position, scale: 1 };
}

function det3(m: Float64Array): number {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);
}

function invert3(m: Mat3): Mat3 {
  const d = det3(Float64Array.from(m));
  const inv = Float32Array.of(
    (m[4] * m[8] - m[5] * m[7]) / d, (m[2] * m[7] - m[1] * m[8]) / d, (m[1] * m[5] - m[2] * m[4]) / d,
    (m[5] * m[6] - m[3] * m[8]) / d, (m[0] * m[8] - m[2] * m[6]) / d, (m[2] * m[3] - m[0] * m[5]) / d,
    (m[3] * m[7] - m[4] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d, (m[0] * m[4] - m[1] * m[3]) / d,
  );
  return inv;
}

export function reprojectionError(K: Mat3, pose: CameraPose, X: Vec3, px: { x: number; y: number }): number {
  const R = pose.R;
  const t = pose.t;
  const xc = R[0] * X.x + R[1] * X.y + R[2] * X.z + t.x;
  const yc = R[3] * X.x + R[4] * X.y + R[5] * X.z + t.y;
  const zc = R[6] * X.x + R[7] * X.y + R[8] * X.z + t.z;
  if (zc >= -1e-9) return Infinity; // 点在相机后方
  const { fx, fy, cx, cy } = mat3ToIntrinsics(K);
  const u = fx * (xc / -zc) + cx;
  const v = fy * (yc / -zc) + cy;
  return Math.hypot(u - px.x, v - px.y);
}

export { normalize3, mat3Mul };
