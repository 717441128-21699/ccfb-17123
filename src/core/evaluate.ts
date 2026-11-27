// 重建精度评测：先做 2D Procrustes 相似对齐（消除坐标系选择差异），再计算五项硬指标
import { bendAxisAt } from './grove';
import type {
  AccuracyReport, GroundTruthCulm, ReconstructedCulm,
} from './types';

export interface Alignment {
  scale: number;
  angle: number;
  tx: number;
  ty: number;
}

/** 用最近邻配对的位置对应求 2D 相似变换（真值 -> 重建坐标系）。
 *  只估计旋转与尺度，平移强制为 0：位置偏差是要评测的指标，不能被对齐吸收。 */
export function procrustesAlign(
  truth: GroundTruthCulm[], recon: ReconstructedCulm[],
): { alignment: Alignment; pairs: { t: GroundTruthCulm; r: ReconstructedCulm }[] } {
  const pairs = nearestNeighborPairs(truth, recon);
  if (pairs.length < 2) {
    return {
      alignment: { scale: 1, angle: 0, tx: 0, ty: 0 },
      pairs,
    };
  }
  const ptsT = pairs.map((p) => p.t.bend.base);
  const ptsR = pairs.map((p) => p.r.bend.base);
  const mt = {
    x: ptsT.reduce((s, p) => s + p.x, 0) / ptsT.length,
    y: ptsT.reduce((s, p) => s + p.y, 0) / ptsT.length,
  };
  const mr = {
    x: ptsR.reduce((s, p) => s + p.x, 0) / ptsR.length,
    y: ptsR.reduce((s, p) => s + p.y, 0) / ptsR.length,
  };
  // 双方去质心后估计旋转与尺度（标准 Procrustes）
  let num = 0;
  let den = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < pairs.length; i++) {
    const ax = ptsT[i].x - mt.x;
    const ay = ptsT[i].y - mt.y;
    const bx = ptsR[i].x - mr.x;
    const by = ptsR[i].y - mr.y;
    num += ax * by - ay * bx;
    den += ax * bx + ay * by;
    sxx += ax * ax + ay * ay;
    syy += bx * bx + by * by;
  }
  const angle = Math.atan2(num, den);
  // 尺度取两侧去质心 RMS 半径之比
  const scale = Math.sqrt(syy / Math.max(1e-12, sxx));
  // 平移让旋转+缩放后的重建质心与真值质心重合
  const mrRot = {
    x: scale * (mr.x * Math.cos(angle) - mr.y * Math.sin(angle)),
    y: scale * (mr.x * Math.sin(angle) + mr.y * Math.cos(angle)),
  };
  const tx = mt.x - mrRot.x;
  const ty = mt.y - mrRot.y;
  return { alignment: { scale, angle, tx, ty }, pairs };
}

export function applyAlignment(p: { x: number; y: number }, a: Alignment): { x: number; y: number } {
  return {
    x: a.scale * (p.x * Math.cos(a.angle) - p.y * Math.sin(a.angle)) + a.tx,
    y: a.scale * (p.x * Math.sin(a.angle) + p.y * Math.cos(a.angle)) + a.ty,
  };
}

function nearestNeighborPairs(
  truth: GroundTruthCulm[], recon: ReconstructedCulm[],
): { t: GroundTruthCulm; r: ReconstructedCulm }[] {
  const usedRecon = new Set<number>();
  const pairs: { t: GroundTruthCulm; r: ReconstructedCulm }[] = [];
  // 以重建结果为基准找最近真值（一根真值最多配一根重建）
  for (const r of recon) {
    let best: GroundTruthCulm | null = null;
    let bestD = Infinity;
    for (const t of truth) {
      const d = Math.hypot(t.bend.base.x - r.bend.base.x, t.bend.base.y - r.bend.base.y);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    if (best && bestD < 0.6 && !usedRecon.has(best.id)) {
      usedRecon.add(best.id);
      pairs.push({ t: best, r });
    }
  }
  return pairs;
}

export function evaluate(
  truth: GroundTruthCulm[], recon: ReconstructedCulm[],
): AccuracyReport {
  const { alignment, pairs } = procrustesAlign(truth, recon);

  // 位置 RMSE
  const posSq = pairs.map(({ t, r }) => {
    const tp = applyAlignment(t.bend.base, alignment);
    return (tp.x - r.bend.base.x) ** 2 + (tp.y - r.bend.base.y) ** 2;
  });
  const positionRmseM = Math.sqrt(avg(posSq));

  // 胸径 RMSE（毫米）
  const diamSq = pairs.map(({ t, r }) => ((t.diameterM - r.diameterM) * 1000) ** 2);
  const diameterRmseMm = Math.sqrt(avg(diamSq));

  // 竹节间距 RMSE：在 1~5 m 高度内采样比较相邻节间距
  const nodeSq: number[] = [];
  for (const { t, r } of pairs) {
    const tt = spacingSeries(t.nodeHeightsM);
    const rr = spacingSeries(r.nodeHeightsM);
    const n = Math.min(tt.length, rr.length);
    for (let i = 0; i < n; i++) nodeSq.push(((tt[i] - rr[i]) * 1000) ** 2);
  }
  const nodeSpacingRmseMm = Math.sqrt(avg(nodeSq));

  // 拆棒：一根真值在 0.25 m 内有多根重建
  let splitErrors = 0;
  for (const t of truth) {
    const tp = applyAlignment(t.bend.base, alignment);
    const near = recon.filter(
      (r) => Math.hypot(r.bend.base.x - tp.x, r.bend.base.y - tp.y) < 0.25,
    );
    if (near.length > 1) splitErrors += near.length - 1;
  }

  // 并棒：一根重建的匹配半径内有两根真值（用未对齐真值粗判）
  let mergeErrors = 0;
  for (const r of recon) {
    const near = truth.filter(
      (t) => Math.hypot(t.bend.base.x - r.bend.base.x, t.bend.base.y - r.bend.base.y)
        < (t.diameterM + r.diameterM) / 2 + 0.05,
    );
    if (near.length > 1) mergeErrors += near.length - 1;
  }

  // 竹龄：±1 年视为正确
  let ageCorrect = 0;
  for (const { t, r } of pairs) {
    if (Math.abs(t.ageYears - r.ageYears) <= 1) ageCorrect++;
  }
  const ageAccuracy = pairs.length ? ageCorrect / pairs.length : 0;

  return {
    positionRmseM,
    diameterRmseMm,
    nodeSpacingRmseMm,
    splitErrors,
    mergeErrors,
    matchedCulms: pairs.length,
    truthCount: truth.length,
    reconCount: recon.length,
    ageAccuracy,
  };
}

function spacingSeries(nodes: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < nodes.length; i++) out.push(nodes[i] - nodes[i - 1]);
  return out;
}

function avg(a: number[]): number {
  return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
}

/** 轨迹评测：相机位置 RMSE（先做相似对齐） */
export function evaluateTrajectory(
  truth: { position: { x: number; y: number; z: number } }[],
  est: { position: { x: number; y: number; z: number } }[],
): number {
  const n = Math.min(truth.length, est.length);
  if (n < 2) return NaN;
  let mt = { x: 0, y: 0, z: 0 };
  let me = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < n; i++) {
    mt = { x: mt.x + truth[i].position.x, y: mt.y + truth[i].position.y, z: mt.z + truth[i].position.z };
    me = { x: me.x + est[i].position.x, y: me.y + est[i].position.y, z: me.z + est[i].position.z };
  }
  mt = { x: mt.x / n, y: mt.y / n, z: mt.z / n };
  me = { x: me.x / n, y: me.y / n, z: me.z / n };
  let st = 0;
  let cross = 0;
  let dot = 0;
  for (let i = 0; i < n; i++) {
    const a = sub(truth[i].position, mt);
    const b = sub(est[i].position, me);
    st += len2(a);
    cross += a.x * b.y - a.y * b.x;
    dot += a.x * b.x + a.y * b.y;
  }
  const s = Math.hypot(cross, dot) / Math.max(1e-12, st);
  const angle = Math.atan2(cross, dot);
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const a = sub(truth[i].position, mt);
    const predicted = {
      x: s * (a.x * Math.cos(angle) - a.y * Math.sin(angle)) + me.x,
      y: s * (a.x * Math.sin(angle) + a.y * Math.cos(angle)) + me.y,
      z: a.z + (me.z - mt.z),
    };
    const b = sub(est[i].position, predicted);
    sq += len2(b);
  }
  return Math.sqrt(sq / n);
}

function sub(a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
function len2(a: { x: number; y: number; z: number }): number {
  return a.x * a.x + a.y * a.y + a.z * a.z;
}

/** 弯曲一致性（评测辅助）：采样若干高度的轴线偏差 */
export function bendDeviationM(
  t: GroundTruthCulm, r: ReconstructedCulm, alignment: Alignment,
): number {
  const heights = [0.5, 1.3, 2.5, 4];
  const ds = heights.map((z) => {
    const tp = applyAlignment(bendAxisAt(t.bend, z), alignment);
    const rp = bendAxisAt(r.bend, z);
    return Math.hypot(tp.x - rp.x, tp.y - rp.y);
  });
  return Math.max(...ds);
}
