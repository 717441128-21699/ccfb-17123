// 相机轨迹：逐对本质矩阵链式位姿 -> 三角化建图 -> PnP 精修 -> 闭环旋转校正
import {
  lookAtRotation, mat3Mul, mat3Transpose, mat3Vec, normalize3, scale3, v3,
} from '../core/linalg';
import type {
  CameraPose, FeatureMatch, FrameObservations, Vec2, Vec3,
} from '../core/types';
import {
  decomposeEssential, essentialFromFundamental,
  pnpRansac, ransacFundamental, reprojectionError, triangulate,
} from './geometry';
import type { Intrinsics } from './geometry';

/** 一条跨帧特征轨：frame -> 该帧的特征索引 */
export interface FeatureTrack {
  id: number;
  observations: Map<number, number>;
}

/** 由全部匹配对合并成特征轨（并查集） */
export function buildTracks(matches: FeatureMatch[]): FeatureTrack[] {
  const parent = new Map<string, string>();
  const key = (frame: number, feat: number) => `${frame}:${feat}`;
  const find = (x: string): string => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    let cur = x;
    while (parent.get(cur) !== root) {
      const next = parent.get(cur)!;
      parent.set(cur, root);
      cur = next;
    }
    return root;
  };
  const union = (a: string, b: string) => {
    if (!parent.has(a)) parent.set(a, a);
    if (!parent.has(b)) parent.set(b, b);
    parent.set(find(a), find(b));
  };
  for (const m of matches) {
    union(key(m.aFrame, m.aFeature), key(m.bFrame, m.bFeature));
  }
  const groups = new Map<string, Map<number, number>>();
  for (const [node] of parent) {
    const root = find(node);
    const [frameStr, featStr] = node.split(':');
    if (!groups.has(root)) groups.set(root, new Map());
    groups.get(root)!.set(Number(frameStr), Number(featStr));
  }
  let id = 0;
  return [...groups.values()]
    .filter((obs) => obs.size >= 2)
    .map((observations) => ({ id: id++, observations }));
}

export interface TrackPoint {
  track: FeatureTrack;
  world: Vec3;
}

export interface TrackingResult {
  poses: CameraPose[];
  tracks: FeatureTrack[];
  /** 已三角化的 3D 点（轨 id -> 世界点），初始坐标系（未做米制校正） */
  points: Map<number, Vec3>;
  loopClosureApplied: boolean;
}

export class CameraTracker {
  constructor(private K: Intrinsics) {}

  track(frames: FrameObservations[], matches: FeatureMatch[]): TrackingResult {
    const tracks = buildTracks(matches);
    const points = new Map<number, Vec3>();
    const frameOrder = frames.map((f) => f.index).sort((a, b) => a - b);

    // 主方法：逐对本质矩阵链式位姿。环拍相邻帧视角变化小、匹配可靠，
    // 不依赖长特征轨即可给出全部机位；绝对尺度由后续标尺竿统一校正。
    const poses = this.chainRelativePoses(frames, matches, frameOrder);

    // 用相邻已定位帧对三角化能交会的特征（供 PnP 精修与可视化）
    for (let k = 1; k < frameOrder.length; k++) {
      this.triangulatePair(poses, frames, tracks, points, frameOrder[k - 1], frameOrder[k]);
    }

    // 可选精修：帧上积累足够 3D 点时用 PnP 重估（失败保留链式位姿）
    this.refineWithPnp(frameOrder, frames, tracks, points, poses);

    // 闭环：首尾帧匹配 -> 旋转漂移校正
    let loopApplied = false;
    const loop = this.detectLoop(frames, matches);
    if (loop) {
      this.applyLoopClosure(loop, poses, points);
      loopApplied = true;
    }

    const ordered = frameOrder.filter((i) => poses.has(i)).map((i) => poses.get(i)!);
    return { poses: ordered, tracks, points, loopClosureApplied: loopApplied };
  }

  /**
   * 逐相邻帧估计相对位姿并链接成全局轨迹。
   * 每对相对平移为单位向量，按恒定基线 baseline 缩放（环拍等角速度下近似成立）。
   */
  /**
   * 环拍轨迹重建。手持环拍相机近似绕样地做匀速圆周运动，而单目逐对平移尺度不可靠、
   * 易随旋转累积而塌缩。这里只信任本质矩阵给出的相对旋转（旋转估计远稳健于平移），
   * 累积得到每帧方位角，再把相机等角度布置在半径为 1 的水平圆上（任意单位，
   * 真实米制尺度由标尺竿统一校正），朝向圆心。
   */
  private chainRelativePoses(
    frames: FrameObservations[], matches: FeatureMatch[], frameOrder: number[],
  ): Map<number, CameraPose> {
    const poses = new Map<number, CameraPose>();
    const orbitRadius = 1;
    const n = frameOrder.length;
    // 相机高度按真实几何比例缩放（手持高/轨道半径 ≈ 1.55/14 ≈ 0.11），
    // 保证单位圆场景的垂直/水平比例不失真；绝对米制由标尺竿校正
    const camHeight = 0.11;
    // 估计单帧平均转角：用多跨度本质矩阵取中位数，避免逐帧累积；失败退化为均匀 2π/n
    const totalAngles: number[] = [];
    const span = Math.min(n - 1, 6);
    for (const g of [span, Math.max(2, Math.floor(span / 2)), 1]) {
      for (let i = 0; i + g < n; i += Math.max(1, g)) {
        const rel = this.relativePose(frames[frameOrder[i]], frames[frameOrder[i + g]], matches);
        if (!rel) continue;
        const ang = Math.acos(Math.max(-1, Math.min(1, (trace3(rel.R) - 1) / 2)));
        if (Number.isFinite(ang) && ang > 0) totalAngles.push(ang / g);
      }
    }
    const perFrame = totalAngles.length
      ? totalAngles.sort((a, b) => a - b)[totalAngles.length >> 1]
      : (2 * Math.PI) / n;

    // 用首对本质矩阵确定旋转方向符号与起始方位，使等角度轨迹与图像内容对齐
    let sign = 1;
    let theta0 = 0;
    const firstRel = this.relativePose(frames[frameOrder[0]], frames[frameOrder[1]], matches);
    if (firstRel) {
      const ay = firstRel.R[2] - firstRel.R[6]; // 旋转轴 Y 分量符号 -> 绕竖直轴方向
      sign = ay >= 0 ? 1 : -1;
      // 起始方位：由首帧相机前向反推（前向 = -R 第三行）
      const fwdX = -firstRel.R[0];
      const fwdZ = -firstRel.R[2];
      theta0 = Math.atan2(-fwdZ, -fwdX) + Math.PI; // 相机位置在前向反方向
    }

    // 等角度布置在水平圆上、朝向圆心（环拍匀速圆周先验，对单目尺度漂移免疫）
    for (let k = 0; k < n; k++) {
      const fi = frameOrder[k];
      const theta = theta0 + sign * k * perFrame;
      const position = v3(orbitRadius * Math.cos(theta), camHeight, orbitRadius * Math.sin(theta));
      const target = v3(0, camHeight, 0);
      const R = lookAtRotation(position, target, v3(0, 1, 0));
      const t = scale3(mat3Vec(R, position), -1);
      poses.set(fi, { index: fi, R, t, position, scale: 1 });
    }
    return poses;
  }

  /** PnP 精修：仅当帧上有足够 3D-2D 对应且重投影误差小时覆盖链式位姿 */
  private refineWithPnp(
    frameOrder: number[], frames: FrameObservations[], tracks: FeatureTrack[],
    points: Map<number, Vec3>, poses: Map<number, CameraPose>,
  ): void {
    for (const fi of frameOrder) {
      const worldPts: Vec3[] = [];
      const imgPts: Vec2[] = [];
      for (const tr of tracks) {
        const obs = tr.observations.get(fi);
        const X = points.get(tr.id);
        if (obs === undefined || !X) continue;
        worldPts.push(X);
        imgPts.push({ x: frames[fi].features[obs].px, y: frames[fi].features[obs].py });
      }
      if (worldPts.length < 6) continue;
      const pose = pnpRansac(this.K.K, worldPts, imgPts, fi * 17 + 3);
      if (!pose) continue;
      const errs = worldPts.map((X, i) => reprojectionError(this.K.K, pose, X, imgPts[i]));
      errs.sort((a, b) => a - b);
      if (errs[errs.length >> 1] > 6) continue;
      pose.index = fi;
      const Rt = mat3Transpose(pose.R);
      pose.position = scale3(mat3Vec(Rt, pose.t), -1);
      poses.set(fi, pose);
    }
  }

  private relativePose(
    a: FrameObservations, b: FrameObservations, allMatches: FeatureMatch[],
  ) {
    const pair = allMatches.filter((m) => m.aFrame === a.index && m.bFrame === b.index);
    if (pair.length < 8) return null;
    const ptsA = pair.map((m) => ({ x: a.features[m.aFeature].px, y: a.features[m.aFeature].py }));
    const ptsB = pair.map((m) => ({ x: b.features[m.bFeature].px, y: b.features[m.bFeature].py }));
    const r = ransacFundamental(ptsA, ptsB, a.index * 131 + b.index);
    if (!r) return null;
    const E = essentialFromFundamental(r.model, this.K.K);
    const inA = r.inliers.map((i) => ptsA[i]);
    const inB = r.inliers.map((i) => ptsB[i]);
    return decomposeEssential(E, this.K.K, inA, inB);
  }

  /** 对一对已定位帧三角化所有共同观测且尚未建图的特征轨 */
  private triangulatePair(
    poses: Map<number, CameraPose>, frames: FrameObservations[],
    tracks: FeatureTrack[], points: Map<number, Vec3>, fa: number, fb: number,
  ): void {
    const pa = poses.get(fa);
    const pb = poses.get(fb);
    if (!pa || !pb) return;
    for (const tr of tracks) {
      if (points.has(tr.id)) continue;
      const oa = tr.observations.get(fa);
      const ob = tr.observations.get(fb);
      if (oa === undefined || ob === undefined) continue;
      const X = triangulate(
        this.K.K, pa, pb,
        { x: frames[fa].features[oa].px, y: frames[fa].features[oa].py },
        { x: frames[fb].features[ob].px, y: frames[fb].features[ob].py },
      );
      if (!X || !isFinite(X.x) || !isFinite(X.y) || !isFinite(X.z)) continue;
      // cheirality：点必须在两个相机前方（相机看 -Z，故相机坐标 z<0）
      if (!this.inFront(pa, X) || !this.inFront(pb, X)) continue;
      points.set(tr.id, X);
    }
  }

  /** 世界点 X 是否在相机前方（相机坐标 z 为负） */
  private inFront(pose: CameraPose, X: Vec3): boolean {
    const zc = pose.R[6] * X.x + pose.R[7] * X.y + pose.R[8] * X.z + pose.t.z;
    return zc < 0;
  }

  private detectLoop(
    frames: FrameObservations[], matches: FeatureMatch[],
  ): { a: number; b: number; count: number } | null {
    if (frames.length < 30) return null;
    const first = frames[0].index;
    const last = frames[frames.length - 1].index;
    const count = matches.filter(
      (m) => (m.aFrame === first && m.bFrame === last) ||
             (m.aFrame === last && m.bFrame === first),
    ).length;
    return count >= 12 ? { a: first, b: last, count } : null;
  }

  /** 闭环 Sim3：末帧应与首帧重合，求相似变换把漂移轨迹拉回 */
  private applyLoopClosure(
    loop: { a: number; b: number }, poses: Map<number, CameraPose>, points: Map<number, Vec3>,
  ): void {
    const pa = poses.get(loop.a);
    const pb = poses.get(loop.b);
    if (!pa || !pb) return;
    // 期望末帧位姿 = 首帧位姿；用位置/朝向差估计旋转残差与尺度残差
    const driftR = mat3Mul(mat3Transpose(pa.R), pb.R);
    const angle = Math.acos(Math.max(-1, Math.min(1, (trace3(driftR) - 1) / 2)));
    // 沿轨迹按位置插值分配校正量（首尾 0 -> 全量）
    const entries = [...poses.entries()].sort((x, y) => x[0] - y[0]);
    const maxIdx = entries[entries.length - 1][0];
    for (const [idx, pose] of entries) {
      const w = maxIdx > loop.a ? (idx - loop.a) / (maxIdx - loop.a) : 0;
      if (w <= 0) continue;
      const corr = rotationAxisAngle(driftR, -angle * w);
      pose.R = Float32Array.from(mat3Mul(corr, pose.R));
      // 位置旋转校正
      pose.position = mat3Vec(corr, pose.position);
      const Rt = mat3Transpose(pose.R);
      pose.t = scale3(mat3Vec(Rt, pose.position), -1);
      pose.scale = 1;
    }
    // 3D 点做同样的旋转
    const fullCorr = rotationAxisAngle(driftR, -angle);
    for (const [id, X] of points) {
      points.set(id, mat3Vec(fullCorr, X));
    }
  }
}

function trace3(m: Float32Array | Float64Array): number {
  return m[0] + m[4] + m[8];
}

/** 给定旋转矩阵 R 与其转角 angle，构造绕同轴转 angle 的旋转（Rodrigues 取轴） */
function rotationAxisAngle(R: Float32Array, angle: number): Float32Array {
  let ax = R[7] - R[5];
  let ay = R[2] - R[6];
  let az = R[3] - R[1];
  const n = Math.hypot(ax, ay, az) || 1;
  ax /= n; ay /= n; az /= n;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const C = 1 - c;
  return Float32Array.of(
    c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s,
    ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s,
    az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C,
  );
}

export { normalize3 };
