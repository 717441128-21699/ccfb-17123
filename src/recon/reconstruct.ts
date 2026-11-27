// 圆柱重建（Web Worker 内运行）：
// 竖棱配对 -> 多视射线交会成竖直圆柱 -> Gauss-Newton 拟合位置/直径/弯曲
// -> 竹节高度聚类 -> 竿色 HSV 竹龄估计 -> 标尺竿尺度校正
import { bendAxisAt } from '../core/grove';
import {
  clamp, lerp, normalize3, solveLeastSquares, sub3, v3,
} from '../core/linalg';
import { meanRgb, rgbToHsv } from '../vision/image';
import type { Intrinsics } from '../vision/geometry';
import type {
  BendModel, CameraPose, FrameObservations, ReconstructedCulm, Vec2, Vec3,
} from '../core/types';

export interface PoleObservation {
  /** 每帧红漆竿的左右棱 x（像素），无观测为 null */
  edges: ({ left: number; right: number; y: number } | null)[];
}

export interface ReconInput {
  frames: FrameObservations[];
  poses: CameraPose[];
  intrinsics: Intrinsics;
  frameImages: ImageData[];
  pole: PoleObservation;
  poleTruthDiameterM: number;
  frameStep: number;
}

interface CulmSeed {
  /** 各帧观测：棱对的中线像素位置与半径（像素） */
  views: { frame: number; cx: number; rPx: number; yTop: number; yBottom: number }[];
}

const WORK_HEIGHT_REF = 360;

export class CylinderReconstructor {
  private poseByFrame = new Map<number, CameraPose>();
  /** 拟合失败诊断（视图数、RMSE），供运行时排查 */
  readonly fitFailures: { views: number; rmse: number }[] = [];

  constructor(private input: ReconInput) {
    for (const p of input.poses) this.poseByFrame.set(p.index, p);
  }

  run(): { culms: ReconstructedCulm[]; poleDiameterEstimate: number; scaleCorrection: number } {
    // 1) 每帧棱配对，按跨帧空间近邻聚成竹竿种子
    const seeds = this.associateCulms();

    // 2) 标尺竿先拟合，得到任意单位 -> 米的尺度
    const poleEstimate = this.estimatePoleDiameter();
    const scaleCorrection = poleEstimate > 1e-9
      ? this.input.poleTruthDiameterM / poleEstimate
      : 1;

    // 3) 每根竹竿 Gauss-Newton 拟合
    const rawCulms: ReconstructedCulm[] = [];
    for (const seed of seeds) {
      const fit = this.fitCylinder(seed);
      if (!fit) continue;
      const nodes = this.estimateNodes(seed, fit.bend, fit.heightRange);
      const { age, confidence, color } = this.estimateAge(seed);
      const diameterM = fit.diameter * scaleCorrection;
      // 物理合理性过滤：真实竹竿胸径约 10~120 mm，剔除离群坏拟合
      if (diameterM < 0.008 || diameterM > 0.15) continue;
      const bend: BendModel = {
        base: { x: fit.bend.base.x * scaleCorrection, y: fit.bend.base.y * scaleCorrection },
        lean: fit.bend.lean,
        bend: {
          x: fit.bend.bend.x / scaleCorrection,
          y: fit.bend.bend.y / scaleCorrection,
        },
      };
      rawCulms.push({
        id: 0,
        bend,
        diameterM,
        diameterSigmaM: fit.diameterSigma * scaleCorrection,
        nodeHeightsM: nodes.map((z) => z * scaleCorrection),
        ageYears: age,
        ageConfidence: confidence,
        meanColor: color,
        observationCount: seed.views.length,
        reprojectionErrorPx: fit.rmsePx,
      });
    }

    // 4) 防止拆/并：过近的重建结果合并，孤立短轨迹剔除，再重新编号
    const merged = this.mergeDuplicates(rawCulms).map((c, i) => ({ ...c, id: i }));
    return {
      culms: merged,
      poleDiameterEstimate: poleEstimate,
      scaleCorrection,
    };
  }

  // ---------- 棱配对与跨帧关联 ----------

  /**
   * 不做脆弱的帧内棱配对，而是把每条竖棱反投影成世界锚点（视线在参考高度的水平交点），
   * 同一根竹竿的左右轮廓与各帧观测在世界系聚集，用空间聚类一次成簇。
   * 簇内按帧合并为带半径的种子视图。
   */
  private associateCulms(): CulmSeed[] {
    const { frames } = this.input;
    interface Anchor {
      frame: number;
      edgeIndex: number;
      cx: number;
      py: number;
      world: Vec2;
    }
    const anchors: Anchor[] = [];
    frames.forEach((frame) => {
      frame.edges.forEach((edge, ei) => {
        const py = (edge.yTop + edge.yBottom) / 2;
        const wp = this.groundPosition(frame.index, edge.xTop, py, 1.3);
        if (wp) {
          anchors.push({ frame: frame.index, edgeIndex: ei, cx: edge.xTop, py, world: wp });
        }
      });
    });
    if (anchors.length < 3) return [];

    // 聚类半径相对相机轨道尺度自适应（链式位姿是任意单位，真值位姿是米制）
    const orbitR = median(
      [...this.poseByFrame.values()].map((p) => Math.hypot(p.position.x, p.position.z)),
    ) || 1;
    const clusterRadius = orbitR * 0.04;
    let clusters: Anchor[][] = [];
    const assign = (list: Anchor[], radius: number) => {
      const out: Anchor[][] = [];
      for (const a of list) {
        let best = -1;
        let bestD = Infinity;
        out.forEach((cl, ci) => {
          const cx = cl.reduce((s, x) => s + x.world.x, 0) / cl.length;
          const cy = cl.reduce((s, x) => s + x.world.y, 0) / cl.length;
          const d = Math.hypot(a.world.x - cx, a.world.y - cy);
          if (d < bestD) {
            bestD = d;
            best = ci;
          }
        });
        if (best >= 0 && bestD < radius) out[best].push(a);
        else out.push([a]);
      }
      return out;
    };
    clusters = assign(anchors, clusterRadius);

    // 多视角射线交会精化：每簇用其全部视线求真实轴线位置，再按到轴线的距离重分配，迭代两轮
    for (let iter = 0; iter < 2; iter++) {
      const refined = clusters.map((cl) => {
        const rays = cl
          .map((a) => this.viewRay(a.frame, a.cx, a.py))
          .filter((r): r is NonNullable<typeof r> => r !== null);
        const center = this.intersectRays(rays, 1.3);
        return center ?? {
          x: cl.reduce((s, a) => s + a.world.x, 0) / cl.length,
          y: cl.reduce((s, a) => s + a.world.y, 0) / cl.length,
        };
      });
      // 重分配：把锚点分给射线轴线最近的簇，但过远时保留原归属，避免碎片化
      const next: Anchor[][] = refined.map(() => []);
      anchors.forEach((a, ai) => {
        let best = -1;
        let bestD = Infinity;
        refined.forEach((c, ci) => {
          const d = Math.hypot(a.world.x - c.x, a.world.y - c.y);
          if (d < bestD) {
            bestD = d;
            best = ci;
          }
        });
        if (best >= 0 && bestD < clusterRadius * 1.5) next[best].push(a);
        else {
          // 回退到初始聚类归属
          const owner = clusters.findIndex((cl) => cl.includes(a));
          if (owner >= 0) next[owner]?.push(a);
          else next[best >= 0 ? best : 0]?.push(a);
        }
        void ai;
      });
      clusters = next.filter((cl) => cl.length > 0);
    }

    // 每个簇转成种子：同一帧的两条棱给出半径，单棱给一个最小半径
    const seeds: CulmSeed[] = [];
    for (const cl of clusters) {
      const framesInCluster = new Map<number, Anchor[]>();
      for (const a of cl) {
        if (!framesInCluster.has(a.frame)) framesInCluster.set(a.frame, []);
        framesInCluster.get(a.frame)!.push(a);
      }
      const views: CulmSeed['views'] = [];
      for (const [frame, group] of framesInCluster) {
        const edgeList = frames[frame]?.edges ?? [];
        const xs = group.map((a) => a.edgeIndex).map((ei) => this.edgeX(edgeList[ei], group[0].py));
        const cx = xs.reduce((s, x) => s + x, 0) / xs.length;
        const rPx = xs.length > 1 ? (Math.max(...xs) - Math.min(...xs)) / 2 : 1.5;
        const yTops = group.map((a) => edgeList[a.edgeIndex]?.yTop ?? 0);
        const yBottoms = group.map((a) => edgeList[a.edgeIndex]?.yBottom ?? 360);
        views.push({
          frame,
          cx,
          rPx: Math.max(1, rPx),
          yTop: Math.min(...yTops),
          yBottom: Math.max(...yBottoms),
        });
      }
      if (views.length >= 2) seeds.push({ views });
    }
    return seeds;
  }

  /**
   * 竖棱的世界水平锚点。单条射线深度欠定，这里给出一个稳定的初始猜测：
   * 沿视线水平方向投到相机轨道半径圆上，后续聚类再用多视角射线交会精化。
   */
  private groundPosition(frame: number, px: number, py: number, _height: number): Vec2 | null {
    const ray = this.viewRay(frame, px, py);
    if (!ray) return null;
    const hn = Math.hypot(ray.dir.x, ray.dir.z);
    if (hn < 1e-7) return null;
    const depth = Math.hypot(ray.origin.x, ray.origin.z) || 1;
    const step = depth / hn;
    return {
      x: ray.origin.x + ray.dir.x * step,
      y: ray.origin.z + ray.dir.z * step,
    };
  }

  private edgeX(e: { xTop: number; xBottom: number; yTop: number; yBottom: number }, y: number): number {
    const t = (y - e.yTop) / Math.max(1, e.yBottom - e.yTop);
    return lerp(e.xTop, e.xBottom, clamp(t, 0, 1));
  }

  // ---------- 圆柱拟合 ----------

  private fitCylinder(seed: CulmSeed): {
    bend: BendModel; diameter: number; diameterSigma: number;
    rmsePx: number; heightRange: [number, number];
  } | null {
    // 射线交会求初始 base（取 y=1.3m 高度附近的视线交点）
    const rays = seed.views.map((v) => ({ view: v, ray: this.viewRay(v.frame, v.cx, (v.yTop + v.yBottom) / 2) }))
      .filter((r) => r.ray);
    if (rays.length < 2) {
      this.fitFailures.push({ views: seed.views.length, rmse: -3 });
      return null;
    }
    const init = this.intersectRays(rays.map((r) => r.ray!), 1.3);
    if (!init) {
      this.fitFailures.push({ views: seed.views.length, rmse: -4 });
      return null;
    }

    // 参数：baseX, baseY, leanX, leanY, bendX, bendY, radius（任意单位）
    // 半径初值完全由像素观测推算，不设大下限（不同位姿尺度下半径可能很小）
    const r0 = median(rays.map((r) => this.pxToRadiusGuess(r.view)));
    const params = new Float64Array([init.x, init.y, 0, 0, 0, 0, clamp(r0, 1e-4, 2)]);

    let rmse = Infinity;
    let prevSse = Infinity;
    for (let iter = 0; iter < 20; iter++) {
      const residuals: number[] = [];
      const Jrows: Float64Array[] = [];
      for (const v of seed.views) {
        const samples = this.samplesOfView(v);
        for (const s of samples) {
          const ray = this.viewRay(v.frame, v.cx, s.y);
          if (!ray) continue;
          const { res, jac } = this.cylinderResidual(params, ray, s.y, v.rPx);
          residuals.push(res);
          Jrows.push(Float64Array.from(jac));
        }
      }
      if (!residuals.length) break;
      const J = new Float64Array(residuals.length * 7);
      const r = new Float64Array(residuals.length);
      Jrows.forEach((row, i) => J.set(row, i * 7));
      residuals.forEach((v, i) => { r[i] = -v; });
      const delta = solveLeastSquares(J, r, residuals.length, 7);
      // 限步长，防止单步越过合理区域
      const maxStep = Math.max(0.5, Math.abs(params[6]));
      for (let i = 0; i < 7; i++) delta[i] = clamp(delta[i], -maxStep, maxStep);
      const candidate = Float64Array.from(params);
      for (let i = 0; i < 7; i++) candidate[i] += delta[i];
      candidate[6] = clamp(candidate[6], 1e-4, 2); // 半径物理边界（任意单位）
      candidate[2] = clamp(candidate[2], -1, 1);
      candidate[3] = clamp(candidate[3], -1, 1);
      candidate[4] = clamp(candidate[4], -0.2, 0.2);
      candidate[5] = clamp(candidate[5], -0.2, 0.2);
      // 接受下降的一步，否则提前终止
      const sse = residuals.reduce((a, x) => a + x * x, 0);
      if (sse >= prevSse && iter > 0) break;
      prevSse = sse;
      params.set(candidate);
      rmse = Math.sqrt(sse / residuals.length);
      if (Math.hypot(...delta) < 1e-7) break;
    }
    if (!isFinite(rmse) || rmse > 8) {
      this.fitFailures.push({
        views: seed.views.length,
        rmse: Number.isFinite(rmse) ? +rmse.toFixed(2) : -1,
      });
      return null;
    }

    const heights = seed.views.flatMap((v) => this.viewHeightRange(v));
    const bend: BendModel = {
      base: { x: params[0], y: params[1] },
      lean: { x: params[2], y: params[3] },
      bend: { x: params[4], y: params[5] },
    };
    const sigma = this.diameterSigma(seed, params);
    return {
      bend,
      diameter: params[6] * 2,
      diameterSigma: sigma * 2,
      rmsePx: rmse,
      heightRange: [Math.min(...heights), Math.max(...heights)],
    };
  }

  private samplesOfView(v: CulmSeed['views'][number]): { y: number }[] {
    const n = 4;
    return Array.from({ length: n }, (_, i) => ({
      y: v.yTop + ((i + 0.5) / n) * (v.yBottom - v.yTop),
    }));
  }

  private viewHeightRange(v: CulmSeed['views'][number]): [number, number] {
    return [this.pixelToHeight(v.frame, v.yBottom), this.pixelToHeight(v.frame, v.yTop)];
  }

  /** 残差：圆柱轮廓半径投影（像素）与观测 rPx 之差；对 7 个参数数值雅可比 */
  private cylinderResidual(
    p: Float64Array,
    ray: { origin: Vec3; dir: Vec3; frameIndex: number },
    yPx: number,
    observedRPx: number,
  ): { res: number; jac: number[] } {
    const predict = (pp: Float64Array) => {
      const bend: BendModel = {
        base: { x: pp[0], y: pp[1] },
        lean: { x: pp[2], y: pp[3] },
        bend: { x: pp[4], y: pp[5] },
      };
      const radius = pp[6];
      const z = this.pixelToHeight(ray.frameIndex, yPx);
      const axis2 = bendAxisAt(bend, z);
      const axis = v3(axis2.x, z, axis2.y);
      // 相机到轴的垂直距离 d，轮廓半宽角 asin(r/d) -> 像素半径
      const oc = sub3(axis, ray.origin);
      const d = Math.hypot(oc.x - ray.dir.x * dot(oc, ray.dir), oc.z - ray.dir.z * dot(oc, ray.dir));
      const ang = Math.asin(clamp(radius / Math.max(radius + 1e-6, d), 0, 0.98));
      return this.input.intrinsics.fx * ang;
    };
    const pred = predict(p);
    const jac: number[] = [];
    const eps = 1e-4;
    for (let i = 0; i < 7; i++) {
      const pp = Float64Array.from(p);
      pp[i] += eps;
      jac.push((predict(pp) - pred) / eps);
    }
    return { res: pred - observedRPx, jac };
  }

  private pxToRadiusGuess(v: CulmSeed['views'][number]): number {
    const ray = this.viewRay(v.frame, v.cx, (v.yTop + v.yBottom) / 2);
    if (!ray) return 0.05;
    // 用射线与 y=1.3 平面交点的行进距离作深度（与链式位姿单位自洽）
    if (Math.abs(ray.dir.y) > 1e-6) {
      const t = (1.3 - ray.origin.y) / ray.dir.y;
      if (t > 0 && isFinite(t)) return (v.rPx / this.input.intrinsics.fx) * t;
    }
    return (v.rPx / this.input.intrinsics.fx) * (ray.depthGuess ?? 5);
  }

  private diameterSigma(seed: CulmSeed, params: Float64Array): number {
    const vals: number[] = [];
    for (const v of seed.views) {
      const ray = this.viewRay(v.frame, v.cx, (v.yTop + v.yBottom) / 2);
      if (!ray) continue;
      const bend: BendModel = {
        base: { x: params[0], y: params[1] },
        lean: { x: params[2], y: params[3] },
        bend: { x: params[4], y: params[5] },
      };
      const z = this.pixelToHeight(v.frame, (v.yTop + v.yBottom) / 2);
      const axis2 = bendAxisAt(bend, z);
      const axis = v3(axis2.x, z, axis2.y);
      const oc = sub3(axis, ray.origin);
      const d = Math.hypot(oc.x - ray.dir.x * dot(oc, ray.dir), oc.z - ray.dir.z * dot(oc, ray.dir));
      vals.push((v.rPx / this.input.intrinsics.fx) * d);
    }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
  }

  // ---------- 射线工具 ----------

  private viewRay(frame: number, px: number, py: number) {
    const pose = this.poseByFrame.get(frame);
    if (!pose) return null;
    const { fx, fy, cx, cy } = this.input.intrinsics;
    // 相机看向 -Z（与 geometry.ts 的 lookAt/三角化约定一致）
    const camDir = normalize3(v3((px - cx) / fx, (py - cy) / fy, -1));
    const Rt = transpose3(pose.R);
    const dir = mat3VecLocal(Rt, camDir);
    const origin = pose.position;
    // 深度初猜：相机到原点平面距离
    const depthGuess = Math.hypot(origin.x, origin.z);
    return { origin, dir, depthGuess, frameIndex: frame };
  }

  /**
   * 多条视线对应的竖直轴线（世界 xz 平面上的 2D 点）。
   * 环拍大基线下用两两水平射线交点的中位数（RANSAC 式稳健），
   * 避免近似平行射线最小二乘的病态。
   */
  private intersectRays(rays: { origin: Vec3; dir: Vec3 }[], _targetHeight: number): Vec2 | null {
    const lines = rays
      .map((r) => {
        const n = Math.hypot(r.dir.x, r.dir.z);
        return n < 1e-7 ? null : {
          ax: r.origin.x, az: r.origin.z,
          ux: r.dir.x / n, uz: r.dir.z / n,
        };
      })
      .filter((l): l is NonNullable<typeof l> => l !== null);
    if (lines.length < 2) return null;
    const xs: number[] = [];
    const zs: number[] = [];
    // 两两求交（采样以控制复杂度）
    for (let i = 0; i < lines.length; i++) {
      for (let j = i + 1; j < lines.length; j++) {
        const a = lines[i];
        const b = lines[j];
        const det = a.ux * b.uz - a.uz * b.ux;
        if (Math.abs(det) < 0.05) continue; // 近平行跳过
        const dx = b.ax - a.ax;
        const dz = b.az - a.az;
        const t = (dx * b.uz - dz * b.ux) / det;
        if (!isFinite(t) || t <= 0) continue;
        xs.push(a.ax + a.ux * t);
        zs.push(a.az + a.uz * t);
      }
    }
    if (xs.length >= 2) return { x: median(xs), y: median(zs) };
    // 近平行导致无有效交点：回退为各射线参考行进距离上点的中位数
    const fbX: number[] = [];
    const fbZ: number[] = [];
    for (const l of lines) {
      const depth = Math.hypot(l.ax, l.az) || 1;
      fbX.push(l.ax + l.ux * depth);
      fbZ.push(l.az + l.uz * depth);
    }
    if (!fbX.length) return null;
    return { x: median(fbX), y: median(fbZ) };
  }

  private pixelToHeight(frame: number, py: number): number {
    // 针孔模型反投影：世界点 y = camY + (cy-py)/fy * depth（相机 -Z 成像约定）
    const pose = this.poseByFrame.get(frame);
    if (!pose) return py / WORK_HEIGHT_REF * 8;
    const { fy, cy } = this.input.intrinsics;
    const depth = Math.hypot(pose.position.x, pose.position.z) * 0.85;
    return pose.position.y + ((cy - py) / fy) * depth;
  }

  // ---------- 竹节 ----------

  private estimateNodes(
    seed: CulmSeed, _bend: BendModel, heightRange: [number, number],
  ): number[] {
    // 收集各帧竹节特征的高度，1D 聚类
    const candidates: number[] = [];
    for (const v of seed.views) {
      const frame = this.input.frames.find((f) => f.index === v.frame);
      if (!frame) continue;
      for (const f of frame.features) {
        if (f.kind !== 'node-ring') continue;
        if (Math.abs(f.px - v.cx) > v.rPx + 6) continue;
        if (f.py < v.yTop - 4 || f.py > v.yBottom + 4) continue;
        candidates.push(this.pixelToHeight(v.frame, f.py));
      }
    }
    candidates.sort((a, b) => a - b);
    const clusters: number[] = [];
    const clusterVals: number[][] = [];
    for (const z of candidates) {
      if (clusters.length && Math.abs(z - clusters[clusters.length - 1]) < 0.12) {
        clusterVals[clusterVals.length - 1].push(z);
        const arr = clusterVals[clusterVals.length - 1];
        clusters[clusters.length - 1] = arr.reduce((a, b) => a + b, 0) / arr.length;
      } else {
        clusters.push(z);
        clusterVals.push([z]);
      }
    }
    // 至少两帧共同观测才采信
    return clusters.filter((_, i) => clusterVals[i].length >= 2)
      .filter((z) => z > heightRange[0] - 0.3 && z < heightRange[1] + 0.3);
  }

  // ---------- 竹龄 ----------

  private estimateAge(
    seed: CulmSeed,
  ): { age: number; confidence: number; color: Vec3 } {
    // 取多个清晰帧的竿面平均色
    let r = 0, g = 0, b = 0, n = 0;
    const scored = seed.views
      .filter((v) => this.input.frameImages[v.frame])
      .sort((a, c) => c.rPx - a.rPx)
      .slice(0, 4);
    for (const v of scored) {
      const img = this.input.frameImages[v.frame];
      const scale = img.width / this.input.intrinsics.cx / 2; // 工作分辨率 -> 原图
      const m = meanRgb(
        img,
        (v.cx - v.rPx * 0.5) * scale, (v.yTop + 4) * scale,
        (v.cx + v.rPx * 0.5) * scale, (v.yBottom - 4) * scale,
      );
      if (!m.n) continue;
      r += m.r; g += m.g; b += m.b; n++;
    }
    if (!n) return { age: 3, confidence: 0.2, color: v3(0.5, 0.5, 0.3) };
    r /= n; g /= n; b /= n;
    const hsv = rgbToHsv(r, g, b);
    // 新竹：色相偏绿（~95°）、高白粉（低饱和高明度）；老竹：色相偏黄（~55°）
    // 线性映射到 1~5 年
    const hueScore = clamp((105 - hsv.h) / 50, 0, 1); // 黄相越多越老
    const powderScore = clamp(1 - hsv.s / 0.6, 0, 1); // 低饱和=白粉=新竹
    const age = clamp(1 + hueScore * 4 - powderScore * 1.2, 1, 5);
    const confidence = clamp(0.45 + 0.1 * n, 0.3, 0.85);
    return {
      age,
      confidence,
      color: { x: r / 255, y: g / 255, z: b / 255 },
    };
  }

  // ---------- 标尺竿 ----------

  private estimatePoleDiameter(): number {
    // 先用各帧视线在参考高度交会，求标尺轴线在任意单位下的水平位置
    const rays: { origin: Vec3; dir: Vec3 }[] = [];
    const samples: { frame: number; rPx: number }[] = [];
    this.input.pole.edges.forEach((e, frameIdx) => {
      if (!e) return;
      const ray = this.viewRay(frameIdx, (e.left + e.right) / 2, e.y);
      if (!ray) return;
      rays.push({ origin: ray.origin, dir: ray.dir });
      samples.push({ frame: frameIdx, rPx: (e.right - e.left) / 2 });
    });
    if (rays.length < 3) return 0;
    let axis = this.intersectRays(rays, 1.3);
    // 退化时用各射线与 y=1.3 平面交点的中位数作为轴线位置
    if (!axis) {
      const ax: number[] = [];
      const az: number[] = [];
      for (const ray of rays) {
        if (Math.abs(ray.dir.y) < 1e-6) continue;
        const t = (1.3 - ray.origin.y) / ray.dir.y;
        if (t > 0 && isFinite(t)) {
          ax.push(ray.origin.x + ray.dir.x * t);
          az.push(ray.origin.z + ray.dir.z * t);
        }
      }
      if (ax.length < 3) return 0;
      axis = { x: median(ax), y: median(az) };
    }
    // 逐帧：相机到竖直轴线的水平距离 d，直径 = 2 * (rPx/fx) * d
    // （竹竿半径远小于观察距离时，视线到轴线垂直距离≈相机到轴水平距离）
    const widths: number[] = [];
    for (const s of samples) {
      const e = this.input.pole.edges[s.frame]!;
      const ray = this.viewRay(s.frame, (e.left + e.right) / 2, e.y);
      if (!ray) continue;
      const d = Math.hypot(ray.origin.x - axis.x, ray.origin.z - axis.y);
      if (d > 1e-3 && isFinite(d)) {
        widths.push(2 * (s.rPx / this.input.intrinsics.fx) * d);
      }
    }
    return widths.length ? median(widths) : 0;
  }

  // ---------- 合并去重 ----------

  private mergeDuplicates(culms: ReconstructedCulm[]): ReconstructedCulm[] {
    // 观测多的优先保留，碎片（同竿过近的重复拟合）并入
    const sorted = culms.slice().sort((a, b) => b.observationCount - a.observationCount);
    const kept: ReconstructedCulm[] = [];
    for (const c of sorted) {
      const dup = kept.find((k) => {
        const d = Math.hypot(c.bend.base.x - k.bend.base.x, c.bend.base.y - k.bend.base.y);
        // 基底间距小于 0.25m 或两者平均半径的 2 倍 -> 同一根被拆成两根
        const rThresh = 2 * (c.diameterM + k.diameterM) / 2;
        return d < Math.max(0.25, rThresh);
      });
      if (!dup) kept.push(c);
    }
    return kept.map((c, i) => ({ ...c, id: i }));
  }
}

function median(arr: number[]): number {
  const a = arr.slice().sort((x, y) => x - y);
  return a[a.length >> 1];
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function transpose3(m: Float32Array): Float32Array {
  return Float32Array.of(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]);
}

function mat3VecLocal(m: Float32Array, v: Vec3): Vec3 {
  return {
    x: m[0] * v.x + m[1] * v.y + m[2] * v.z,
    y: m[3] * v.x + m[4] * v.y + m[5] * v.z,
    z: m[6] * v.x + m[7] * v.y + m[8] * v.z,
  };
}

export { normalize3, sub3 };
