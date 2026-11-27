// 竹林特征提取（CPU 回退路径，与 WebGPU 内核同算法）：
// 1) 竖直棱线（竹竿左右轮廓）2) 竹节环（水平暗带）3) 竿面斑痕（局部暗点）
// 风动区域由帧差运动掩码剔除，描述子为 4x4 x 8 方向的 SIFT 风格 128 维
import type { CulmFeature, FrameObservations, VerticalEdge } from '../core/types';
import type { GradImage, GrayImage } from './image';
import {
  dilate, motionMask, resizeGray, sobel, toGray,
} from './image';

const DESC_BINS = 8;
const DESC_GRID = 4;
export const DESCRIPTOR_DIM = DESC_BINS * DESC_GRID * DESC_GRID; // 128

export interface ExtractionOptions {
  workW: number;
  workH: number;
  /** 棱的最小长度（工作分辨率像素） */
  minEdgeLen: number;
  /** 运动掩码膨胀半径 */
  motionDilate: number;
}

export const DEFAULT_EXTRACTION: ExtractionOptions = {
  workW: 640,
  workH: 360,
  minEdgeLen: 60,
  motionDilate: 3,
};

export class FeatureExtractor {
  private prevGray: GrayImage | null = null;
  private opts: ExtractionOptions;

  constructor(opts: Partial<ExtractionOptions> = {}) {
    this.opts = { ...DEFAULT_EXTRACTION, ...opts };
  }

  extract(index: number, frame: ImageData, timeSec: number): FrameObservations {
    const full = toGray(frame);
    const gray = full.w === this.opts.workW ? full : resizeGray(full, this.opts.workW, this.opts.workH);
    const { w, h } = gray;

    // 运动掩码：风动竹叶与手持抖动区域不参与匹配
    let mask: Uint8Array = new Uint8Array(w * h);
    let motionRatio = 0;
    if (this.prevGray) {
      const mm = motionMask(this.prevGray, gray, 14, this.opts.motionDilate);
      mask = mm.mask;
      motionRatio = mm.ratio;
    }
    this.prevGray = gray;

    const grad = sobel(gray);
    const edges = this.detectVerticalEdges(gray, grad, mask);
    const features: CulmFeature[] = [];
    for (let ei = 0; ei < edges.length; ei++) {
      const nodes = this.detectNodeRings(gray, grad, edges[ei], mask);
      const spots = this.detectSpots(gray, grad, edges[ei], mask);
      for (const p of [...nodes, ...spots]) {
        const descriptor = this.descriptor(grad, p.x, p.y, p.scale, mask);
        if (descriptor) {
          features.push({
            px: p.x, py: p.y, kind: p.kind, descriptor,
            edgeIndex: ei, scale: p.scale,
          });
        }
      }
    }
    // 天空掩码（高亮度、低梯度的顶部区域占比，统计用）
    let sky = 0;
    for (let y = 0; y < h; y += 4) {
      for (let x = 0; x < w; x += 4) {
        const i = y * w + x;
        if (gray.data[i] > 175 && grad.mag[i] < 12) sky++;
      }
    }
    const skyRatio = sky / ((w / 4) * (h / 4));
    return {
      index, edges, features, motionMaskRatio: motionRatio, skyMaskRatio: skyRatio, timeSec,
    };
  }

  // ---------- 竖直棱线 ----------

  private detectVerticalEdges(gray: GrayImage, grad: GradImage, mask: Uint8Array): VerticalEdge[] {
    const { w, h } = gray;
    // 竖向边缘能量：|gx| 强、gy 弱（即梯度近水平，棱线竖直）
    const score = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const gx = grad.gx[i];
      const gy = grad.gy[i];
      const verticality = Math.abs(gx) / (Math.abs(gy) + 1);
      score[i] = verticality > 2.2 ? Math.abs(gx) : 0;
    }
    // 按列做非极大值抑制，每列取局部峰（一条竿左右两条棱，占不同列）
    const colPeak = new Int16Array(w * h).fill(-1);
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const i = y * w + x;
        if (mask[i] || score[i] < 30) continue;
        let peak = true;
        for (let dx = -2; dx <= 2; dx++) {
          if (dx !== 0 && score[y * w + x + dx] > score[i]) { peak = false; break; }
        }
        if (peak) colPeak[i] = 1;
      }
    }
    // 沿列向下生长为线段，允许小的横向漂移（亚像素中心）
    const visited = new Uint8Array(w * h);
    const edges: VerticalEdge[] = [];
    for (let x = 1; x < w - 1; x++) {
      for (let y = 1; y < h - 1; y++) {
        if (colPeak[y * w + x] !== 1 || visited[y * w + x]) continue;
        const pts: { x: number; y: number; s: number }[] = [];
        let cx = x;
        let cy = y;
        while (cy < h - 1) {
          let bestX = -1;
          let bestS = 0;
          for (let dx = -2; dx <= 2; dx++) {
            const xx = Math.round(cx) + dx;
            if (xx < 1 || xx >= w - 1) continue;
            const idx = cy * w + xx;
            if (colPeak[idx] === 1 && !visited[idx] && score[idx] > bestS) {
              bestS = score[idx];
              bestX = xx;
            }
          }
          if (bestX < 0) break;
          visited[cy * w + bestX] = 1;
          // 亚像素 x：用邻域 score 加权
          const i0 = cy * w + bestX;
          const sl = score[i0 - 1] || 0;
          const sr = score[i0 + 1] || 0;
          const sub = (sr - sl) / (2 * (sl + score[i0] + sr + 1e-6));
          pts.push({ x: bestX + sub, y: cy, s: bestS });
          cx = bestX;
          cy++;
        }
        if (pts.length >= this.opts.minEdgeLen) {
          const top = pts[0];
          const bot = pts[pts.length - 1];
          const avgS = pts.reduce((a, p) => a + p.s, 0) / pts.length;
          edges.push({
            xTop: top.x, xBottom: bot.x, yTop: top.y, yBottom: bot.y,
            strength: Math.min(1, avgS / 160),
          });
        }
      }
    }
    // 合并间距过小、方向一致的平行重复棱（同一轮廓的双响应）
    return this.mergeEdges(edges);
  }

  private mergeEdges(edges: VerticalEdge[]): VerticalEdge[] {
    const sorted = edges.slice().sort((a, b) => a.xTop - b.xTop);
    const out: VerticalEdge[] = [];
    for (const e of sorted) {
      const last = out[out.length - 1];
      if (last && Math.abs(e.xTop - last.xTop) < 3 &&
          Math.abs(e.yTop - last.yTop) < 12 &&
          Math.abs(e.yBottom - last.yBottom) < 12) {
        // 合并为更强的一条
        if (e.strength > last.strength) out[out.length - 1] = e;
      } else {
        out.push(e);
      }
    }
    return out;
  }

  // ---------- 竹节环：棱线附近的水平暗带 ----------

  private detectNodeRings(
    gray: GrayImage, grad: GradImage, edge: VerticalEdge, mask: Uint8Array,
  ): { x: number; y: number; scale: number; kind: 'node-ring' }[] {
    const { w } = gray;
    const out: { x: number; y: number; scale: number; kind: 'node-ring' }[] = [];
    const bandW = 5;
    let lastNodeY = -1e9;
    for (let y = edge.yTop + 8; y < edge.yBottom - 8; y += 2) {
      const xc = this.edgeXAt(edge, y);
      // 水平梯度强 + 局部亮度低谷 = 竹节环
      let horiz = 0;
      let dark = 0;
      let n = 0;
      for (let dx = -bandW; dx <= bandW; dx++) {
        const xx = Math.round(xc) + dx;
        if (xx < 1 || xx >= w - 1 || mask[y * w + xx]) continue;
        horiz += Math.abs(grad.gy[y * w + xx]);
        dark += gray.data[y * w + xx];
        n++;
      }
      if (!n) continue;
      dark /= n;
      const above = this.meanBand(gray, edge, y - 5, mask);
      const below = this.meanBand(gray, edge, y + 5, mask);
      const dip = (above + below) / 2 - dark;
      if (horiz / n > 26 && dip > 7 && y - lastNodeY > 14) {
        out.push({ x: xc, y, scale: 6, kind: 'node-ring' });
        lastNodeY = y;
      }
    }
    return out;
  }

  // ---------- 斑痕：棱线内侧的局部暗点 ----------

  private detectSpots(
    gray: GrayImage, grad: GradImage, edge: VerticalEdge, mask: Uint8Array,
  ): { x: number; y: number; scale: number; kind: 'spot' }[] {
    const { w } = gray;
    const out: { x: number; y: number; scale: number; kind: 'spot' }[] = [];
    let lastY = -1e9;
    for (let y = edge.yTop + 10; y < edge.yBottom - 10; y += 6) {
      const xc = this.edgeXAt(edge, y);
      const i = y * w + Math.round(xc);
      if (mask[i]) continue;
      const local = this.meanBand(gray, edge, y, mask, 3);
      const dip = local - gray.data[i];
      if (dip > 10 && grad.mag[i] > 18 && y - lastY > 16) {
        out.push({ x: xc, y, scale: 4, kind: 'spot' });
        lastY = y;
      }
    }
    return out;
  }

  private edgeXAt(e: VerticalEdge, y: number): number {
    const t = (y - e.yTop) / Math.max(1, e.yBottom - e.yTop);
    return e.xTop + t * (e.xBottom - e.xTop);
  }

  private meanBand(gray: GrayImage, e: VerticalEdge, y: number, mask: Uint8Array, halfW = 5): number {
    if (y < 0 || y >= gray.h) return 0;
    const xc = this.edgeXAt(e, y);
    let s = 0;
    let n = 0;
    for (let dx = -halfW; dx <= halfW; dx++) {
      const xx = Math.round(xc) + dx;
      if (xx < 0 || xx >= gray.w || mask[y * gray.w + xx]) continue;
      s += gray.data[y * gray.w + xx];
      n++;
    }
    return n ? s / n : 0;
  }

  // ---------- SIFT 风格描述子 ----------

  private descriptor(
    grad: GradImage, px: number, py: number, scale: number, mask: Uint8Array,
  ): Float32Array | null {
    const { w, h, mag, ori } = grad;
    const radius = scale * 3;
    if (px - radius < 2 || px + radius >= w - 2 || py - radius < 2 || py + radius >= h - 2) {
      return null;
    }
    // 主方向：竹节环/斑痕附近梯度多为水平，主方向取局部加权直方图峰
    const hist = new Float32Array(36);
    for (let dy = -radius; dy <= radius; dy += 2) {
      for (let dx = -radius; dx <= radius; dx += 2) {
        const x = Math.round(px + dx);
        const y = Math.round(py + dy);
        if (mask[y * w + x]) continue;
        const wt = mag[y * w + x] * Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
        const bin = Math.floor((ori[y * w + x] / (Math.PI * 2)) * 36) % 36;
        hist[bin] += wt;
      }
    }
    let mainBin = 0;
    for (let b = 1; b < 36; b++) if (hist[b] > hist[mainBin]) mainBin = b;
    const mainOri = (mainBin / 36) * Math.PI * 2;

    const desc = new Float32Array(DESCRIPTOR_DIM);
    const cell = (radius * 2) / DESC_GRID;
    for (let gy = 0; gy < DESC_GRID; gy++) {
      for (let gx = 0; gx < DESC_GRID; gx++) {
        const cx = px - radius + (gx + 0.5) * cell;
        const cy = py - radius + (gy + 0.5) * cell;
        for (let dy = -cell / 2; dy <= cell / 2; dy += 2) {
          for (let dx = -cell / 2; dx <= cell / 2; dx += 2) {
            const x = Math.round(cx + dx);
            const y = Math.round(cy + dy);
            if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1 || mask[y * w + x]) continue;
            let a = ori[y * w + x] - mainOri;
            a = ((a % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
            const bin = Math.min(DESC_BINS - 1, Math.floor((a / (Math.PI * 2)) * DESC_BINS));
            desc[(gy * DESC_GRID + gx) * DESC_BINS + bin] += mag[y * w + x];
          }
        }
      }
    }
    // 归一化 + 截断（SIFT 标准后处理，抗光照变化）
    let norm = 0;
    for (const v of desc) norm += v * v;
    norm = Math.sqrt(norm) + 1e-9;
    for (let i = 0; i < desc.length; i++) desc[i] = Math.min(0.2, desc[i] / norm);
    let norm2 = 0;
    for (const v of desc) norm2 += v * v;
    norm2 = Math.sqrt(norm2) + 1e-9;
    for (let i = 0; i < desc.length; i++) desc[i] /= norm2;
    return desc;
  }
}

export { dilate };
