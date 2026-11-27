// 帧间特征匹配：描述子最近邻 + Lowe 比率 + 同类约束 + 竖序一致性
import type { FeatureMatch, FrameObservations } from '../core/types';
import { DESCRIPTOR_DIM } from './features';

export interface MatchOptions {
  ratioThreshold: number;
  maxDistance: number;
  /** 竖序违例的最大比例 */
  maxOrderViolation: number;
}

export const DEFAULT_MATCH: MatchOptions = {
  ratioThreshold: 0.78,
  maxDistance: 0.9,
  maxOrderViolation: 0.25,
};

function descDistance(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < DESCRIPTOR_DIM; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export function matchFrames(
  a: FrameObservations, b: FrameObservations, opts: Partial<MatchOptions> = {},
): FeatureMatch[] {
  const o = { ...DEFAULT_MATCH, ...opts };
  const matches: FeatureMatch[] = [];
  // 按类型分组匹配，竹节环优先
  for (const kind of ['node-ring', 'spot'] as const) {
    const fa = a.features.map((f, i) => ({ f, i })).filter((x) => x.f.kind === kind);
    const fb = b.features.map((f, i) => ({ f, i })).filter((x) => x.f.kind === kind);
    if (!fa.length || !fb.length) continue;
    for (const pa of fa) {
      let best = Infinity;
      let second = Infinity;
      let bestJ = -1;
      for (const pb of fb) {
        const d = descDistance(pa.f.descriptor, pb.f.descriptor);
        if (d < best) {
          second = best;
          best = d;
          bestJ = pb.i;
        } else if (d < second) {
          second = d;
        }
      }
      if (bestJ < 0 || best > o.maxDistance) continue;
      if (best >= o.ratioThreshold * second) continue;
      matches.push({
        aFrame: a.index, aFeature: pa.i,
        bFrame: b.index, bFeature: bestJ,
        distance: best,
      });
    }
  }
  return enforceVerticalOrder(a, b, matches, o.maxOrderViolation);
}

/** 竖序一致性：真实竹竿上的特征在相邻帧保持上下顺序，剔除明显违例 */
function enforceVerticalOrder(
  a: FrameObservations, b: FrameObservations, matches: FeatureMatch[], maxViolation: number,
): FeatureMatch[] {
  if (matches.length < 4) return matches;
  // 按左帧 y 排序后，右帧 y 也应基本单调
  const sorted = matches.slice().sort(
    (m1, m2) => a.features[m1.aFeature].py - a.features[m2.aFeature].py,
  );
  let violations = 0;
  for (let i = 1; i < sorted.length; i++) {
    const yPrev = b.features[sorted[i - 1].bFeature].py;
    const yCur = b.features[sorted[i].bFeature].py;
    if (yCur < yPrev - 6) violations++;
  }
  if (violations / sorted.length <= maxViolation) return matches;
  // 违例过多：只保留满足单调的最长子序列
  const n = sorted.length;
  const dp = new Array(n).fill(1);
  const prev = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (b.features[sorted[j].bFeature].py <= b.features[sorted[i].bFeature].py + 6 && dp[j] + 1 > dp[i]) {
        dp[i] = dp[j] + 1;
        prev[i] = j;
      }
    }
  }
  let end = 0;
  for (let i = 1; i < n; i++) if (dp[i] > dp[end]) end = i;
  const keep = new Set<number>();
  let cur = end;
  while (cur >= 0) {
    keep.add(cur);
    cur = prev[cur];
  }
  return sorted.filter((_, i) => keep.has(i));
}

/** 相邻帧序列匹配（含隔帧），输出所有匹配对 */
export function matchSequence(
  frames: FrameObservations[], gap = 2, opts: Partial<MatchOptions> = {},
): FeatureMatch[] {
  const all: FeatureMatch[] = [];
  for (let i = 0; i < frames.length; i++) {
    for (let g = 1; g <= gap && i + g < frames.length; g++) {
      all.push(...matchFrames(frames[i], frames[i + g], opts));
    }
  }
  return all;
}
